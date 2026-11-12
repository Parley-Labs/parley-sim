import asyncio
import json
import logging
import time

import pytest

pytest.importorskip("openenv")

from fastapi.testclient import TestClient
from openenv.core.env_server.serialization import serialize_observation

from parley_sim.adapters.openenv import (
    MAX_ACTION_CHARS,
    ParleySimAction,
    ParleySimEnv,
    ParleySimEnvironment,
    create_parley_app,
)
from parley_sim.drivers.models import ModelSpec
from parley_sim.envs.support_desk import SupportDesk, make_policy
from parley_sim.envs.support_desk.metrics import episode_metrics
from parley_sim.runtime import verify

SCRIPTED = {"agent": "client", "customer": "policy:reasonable"}


def play(env, seed=7):
    """Drive the focal agent with the scripted agent policy, as structured moves."""
    policy = make_policy("agent_scripted", "agent")
    obs = env.reset(seed=seed)
    steps = 0
    while not obs.done:
        assert obs.seat == "agent" and obs.system_prompt and obs.action_schema
        obs = env.step(ParleySimAction(move=policy.decide(obs.observation)))
        steps += 1
    return obs, steps


def test_scripted_episode_replays_and_reward_is_the_report_metric():
    env = ParleySimEnvironment(seats=SCRIPTED)
    obs, steps = play(env)
    state = env.state
    assert state.done and state.step_count == steps and state.seed == 7
    assert verify(SupportDesk(), state.trace) == []
    assert obs.reward == episode_metrics(state.trace)["focal_surplus_share"]
    assert obs.outcomes == state.trace[-1]["outcomes"]
    assert {l["seat"] for l in state.trace if l["type"] == "decision"} == {"agent", "customer"}


def test_malformed_and_illegal_actions_are_strikes_not_crashes():
    env = ParleySimEnvironment(seats=SCRIPTED)
    env.reset(seed=3)
    obs = env.step(ParleySimAction(text="not json"))
    assert obs.last_strike == "malformed_output" and not obs.done
    bad = {"message": "", "move": {"type": "propose", "deal": {"discount_pct": 15}}}
    obs = env.step(ParleySimAction(text=json.dumps(bad)))
    assert obs.last_strike == "illegal_action"
    ok = {"message": "hi", "move": {"type": "none"}, "explanation": "wait"}
    obs = env.step(ParleySimAction(text=json.dumps(ok)))
    assert obs.last_strike is None
    focal = [l for l in env.state.trace if l["type"] == "decision" and l["seat"] == "agent"]
    assert [d["strike"] for d in focal] == ["malformed_output", "illegal_action", None]
    assert focal[2]["explanation"] == "wait"


class Walker:
    async def complete(self, *, system, user, schema):
        reply = {"message": "I'm leaving.", "move": {"type": "walk_away"}, "explanation": "no"}
        return json.dumps(reply), {"usage": {"input_tokens": 3, "output_tokens": 2}}


def test_model_counterpart_runs_inside_the_environment():
    env = ParleySimEnvironment(
        seats={"agent": "client", "customer": "model:peer"},
        lineup=[{"name": "peer", "provider": "openai_compat", "model": "x"}],
        client_factory=lambda spec: Walker())
    obs = env.reset(seed=1)
    assert obs.done and obs.metrics["deal"] == 0.0  # the customer opened by walking away
    customer = [l for l in env.state.trace if l["type"] == "decision" and l["seat"] == "customer"]
    assert customer[0]["source"] == "external" and customer[0]["raw"]["usage"]["input_tokens"] == 3


def test_seats_must_name_the_client_and_step_needs_reset():
    with pytest.raises(ValueError):
        ParleySimEnvironment(seats={"agent": "policy:agent_scripted", "customer": "policy:hardliner"})
    with pytest.raises(RuntimeError):
        ParleySimEnvironment(seats=SCRIPTED).step(ParleySimAction(move={"move": {"type": "none"}}))
    with pytest.raises(ValueError):
        ParleySimAction(text="x", move={})


def test_client_parses_server_payloads():
    env = ParleySimEnvironment(seats=SCRIPTED)
    payload = serialize_observation(env.reset(seed=2))
    result = ParleySimEnv(base_url="http://unused")._parse_result(payload)
    assert result.observation.seat == "agent" and not result.done
    state = ParleySimEnv(base_url="http://unused")._parse_state(env.state.model_dump())
    assert state.seed == 2 and state.trace


def test_served_app_runs_an_episode_over_websocket():
    app = create_parley_app(seats=SCRIPTED)
    client = TestClient(app)
    policy = make_policy("agent_scripted", "agent")
    with client.websocket_connect("/ws") as ws:
        ws.send_text(json.dumps({"type": "reset", "data": {"seed": 5}}))
        msg = ws.receive_json()
        assert msg["type"] == "observation", msg
        data = msg["data"]
        while not data["done"]:
            move = policy.decide(data["observation"]["observation"])
            ws.send_text(json.dumps({"type": "step", "data": {"move": move}}))
            msg = ws.receive_json()
            assert msg["type"] == "observation", msg
            data = msg["data"]
        ws.send_text(json.dumps({"type": "state"}))
        state = ws.receive_json()["data"]
    assert data["reward"] == episode_metrics(state["trace"])["focal_surplus_share"]
    assert verify(SupportDesk(), state["trace"]) == []


def test_state_hides_private_information_until_the_episode_ends():
    env = ParleySimEnvironment(seats=SCRIPTED)
    env.reset(seed=7)
    trace = env.state.trace
    assert all(l["type"] == "event" or l.get("seat") == "agent" for l in trace)
    assert not any(l["type"] in ("header", "footer") for l in trace)
    assert "walkaway" not in json.dumps(trace)
    play(env)
    assert env.state.trace[0]["type"] == "header"


@pytest.mark.parametrize("kwargs,match", [
    ({"reward_metric": "focal_surplus"}, "reward_metric"),
    ({"seats": {"agent": "client", "customer": "polcy:reasonable"}}, "must start with"),
    ({"seats": {"agent": "client", "customer": "model:peer"}}, "not in the lineup"),
    ({"seats": {"agent": "client", "customer": "policy:nobody"}}, "unknown policy"),
    ({"release_trace": "sometimes"}, "release_trace"),
])
def test_misconfiguration_fails_at_startup(kwargs, match):
    with pytest.raises(ValueError, match=match):
        ParleySimEnvironment(**{"seats": SCRIPTED, **kwargs})


def test_focal_decisions_mid_episode_carry_no_evaluation_context():
    env = ParleySimEnvironment(seats={"agent": "client", "customer": "policy:hardliner"})
    obs = env.reset(seed=11)
    policy = make_policy("agent_scripted", "agent")
    while not obs.done and not any(l["type"] == "decision" for l in env.state.trace):
        obs = env.step(ParleySimAction(move=policy.decide(obs.observation)))
    assert not obs.done  # seed 11: the customer's offer is still standing
    mid = env.state.trace
    assert any(l["type"] == "decision" for l in mid)
    assert all("context" not in l for l in mid)
    assert "counterpart_value" not in json.dumps(mid)


def test_closed_session_refuses_further_episodes():
    env = ParleySimEnvironment(seats=SCRIPTED)
    env.reset(seed=1)
    env.close()
    with pytest.raises(RuntimeError, match="closed"):
        env.reset(seed=2)


class FakeSDK:
    """Stands in for AsyncOpenAI/AsyncAnthropic: records close()."""

    def __init__(self):
        self.closed = 0

    async def close(self):
        self.closed += 1


class Recording:
    """A parley-sim provider client: the SDK client it wraps is `_client`."""

    def __init__(self, delay=0.0):
        self._client = FakeSDK()
        self.calls = 0
        self.delay = delay

    async def complete(self, *, system, user, schema):
        self.calls += 1
        await asyncio.sleep(self.delay)
        reply = {"message": "Hello.", "move": {"type": "none"}, "explanation": "wait"}
        return json.dumps(reply), {"usage": {"input_tokens": 1, "output_tokens": 1}}


def recording_factory(made, **kwargs):
    def make(spec):
        made.append(Recording(**kwargs))
        return made[-1]
    return make


PEER = {"seats": {"agent": "client", "customer": "model:peer"},
        "lineup": [{"name": "peer", "provider": "openai_compat", "model": "x", "timeout_s": 5}]}
OK_REPLY = json.dumps({"message": "hi", "move": {"type": "none"}, "explanation": "wait"})


def focal_decisions(env):
    return [l for l in env.state.trace if l["type"] == "decision" and l["seat"] == "agent"]


def test_stateless_http_routes_run_no_episode_and_make_no_clients():
    made = []
    app = create_parley_app(**PEER, client_factory=recording_factory(made))
    assert len(made) == 1 and made[0]._client.closed == 1  # the startup check, closed again
    client = TestClient(app)
    for method, path, body in [("POST", "/reset", {"seed": 5}),
                               ("POST", "/step", {"action": {"text": OK_REPLY}}),
                               ("GET", "/state", None),
                               ("POST", "/mcp", {"jsonrpc": "2.0", "id": 1,
                                                 "method": "tools/list"})]:
        resp = client.request(method, path, json=body)
        assert resp.status_code == 405, (path, resp.text)
        assert "/ws" in resp.json()["detail"]
    assert len(made) == 1 and made[0].calls == 0
    assert client.get("/health").status_code == 200
    assert not {"/reset", "/step", "/state"} & set(client.get("/openapi.json").json()["paths"])


def test_web_interface_is_never_mounted(monkeypatch, caplog):
    monkeypatch.setenv("ENABLE_WEB_INTERFACE", "true")
    with caplog.at_level(logging.WARNING, logger="parley_sim.adapters.openenv"):
        app = create_parley_app(seats=SCRIPTED)
    assert "openenv push --no-interface" in caplog.text
    assert not any(getattr(r, "path", "").startswith("/web") for r in app.routes)
    assert TestClient(app).get("/web").status_code == 404


def test_release_trace_never_withholds_counterpart_information_after_done():
    env = ParleySimEnvironment(seats=SCRIPTED, release_trace="never")
    obs, _ = play(env)
    state = env.state
    assert state.done and state.outcomes == obs.outcomes and state.metrics == obs.metrics
    assert state.trace and all(l["type"] == "event" or l.get("seat") == "agent" for l in state.trace)
    assert not any(l["type"] in ("header", "footer") for l in state.trace)
    assert all("context" not in l for l in state.trace)
    assert "walkaway" not in json.dumps(state.trace)
    released = ParleySimEnvironment(seats=SCRIPTED)  # the default releases it, same seed
    play(released)
    assert released.state.trace[0]["type"] == "header"


def test_any_parse_failure_is_a_strike(monkeypatch):
    env = ParleySimEnvironment(seats=SCRIPTED)
    env.reset(seed=3)
    deep = "[" * 30000 + "]" * 30000  # under the size limit, over the recursion limit
    obs = env.step(ParleySimAction(text=deep))
    assert obs.last_strike == "malformed_output" and not obs.done

    # Whether that nesting overflows depends on the Python version (3.14's json
    # parser copes), so raise each failure type directly as well.
    for exc in (RecursionError("too deep"), TypeError("odd payload")):
        def fail(payload, exc=exc):
            raise exc
        monkeypatch.setattr(env.text, "parse", fail)
        assert env.step(ParleySimAction(text=OK_REPLY)).last_strike == "malformed_output"
        monkeypatch.undo()
    assert env.step(ParleySimAction(text=OK_REPLY)).last_strike is None
    details = [d["detail"] for d in focal_decisions(env)]
    assert details[1:] == ["RecursionError: too deep", "TypeError: odd payload", None]
    assert env.state.step_count == 4


def test_unexpected_step_error_keeps_the_request_pending(monkeypatch):
    env = ParleySimEnvironment(seats=SCRIPTED)
    env.reset(seed=3)

    def bug(payload):
        raise KeyError("bug")
    monkeypatch.setattr(env.text, "parse", bug)
    with pytest.raises(KeyError):
        env.step(ParleySimAction(text=OK_REPLY))
    monkeypatch.undo()
    obs = env.step(ParleySimAction(text=OK_REPLY))
    assert obs.last_strike is None and not obs.done and env.state.step_count == 1
    assert len(focal_decisions(env)) == 1


def test_bad_reset_arguments_leave_the_running_episode_intact():
    env = ParleySimEnvironment(seats=SCRIPTED)
    obs = env.reset(seed=1, episode_id="ep-1")
    with pytest.raises(ValueError):
        env.reset(seed=2, episode_id=123)  # not a string: the state model rejects it
    with pytest.raises(ValueError):
        env.reset(seed="not a seed")
    assert env.state.seed == 1 and env.state.episode_id == "ep-1"
    policy = make_policy("agent_scripted", "agent")
    while not obs.done:
        obs = env.step(ParleySimAction(move=policy.decide(obs.observation)))
    trace = env.state.trace
    assert verify(SupportDesk(), trace) == [] and trace[0]["spec"]["seed"] == 1


def test_client_waits_for_two_counterpart_attempts():
    default_timeout = ModelSpec(name="m", provider="openai", model="x").timeout_s
    client = ParleySimEnv(base_url="http://unused")
    assert client._message_timeout == 300 and client._message_timeout > 2 * default_timeout
    assert ParleySimEnv(base_url="http://unused", message_timeout_s=30)._message_timeout == 30


def test_session_closed_during_a_counterpart_call_returns_no_observation():
    made = []
    env = ParleySimEnvironment(**PEER, client_factory=recording_factory(made, delay=0.2))

    async def main():
        task = asyncio.create_task(env.reset_async(seed=1))  # the customer opens: a slow call
        await asyncio.sleep(0.05)
        env.close()  # what OpenEnv's idle reaper does
        with pytest.raises(RuntimeError, match="closed"):
            await task
        await asyncio.sleep(0)
    asyncio.run(main())
    assert made[0].calls == 1 and made[0]._client.closed == 1


def test_session_timeout_must_outlast_a_counterpart_decision():
    slow = {**PEER, "lineup": [{**PEER["lineup"][0], "timeout_s": 100}]}
    with pytest.raises(ValueError, match="PARLEY_SESSION_TIMEOUT"):
        create_parley_app(**slow, client_factory=recording_factory([]), session_timeout_s=150)
    create_parley_app(**slow, client_factory=recording_factory([]), session_timeout_s=200)
    create_parley_app(**slow, client_factory=recording_factory([]), session_timeout_s=None)
    create_parley_app(seats=SCRIPTED, session_timeout_s=1)  # no model calls to outlast


def test_close_closes_provider_clients_once():
    made = []
    env = ParleySimEnvironment(**PEER, client_factory=recording_factory(made))
    env.reset(seed=1)
    env.close()
    env.close()
    assert made[0].calls == 1 and made[0]._client.closed == 1


def test_served_sessions_close_their_clients():
    made = []
    app = create_parley_app(**PEER, client_factory=recording_factory(made))
    with TestClient(app) as client:
        with client.websocket_connect("/ws") as ws:
            ws.send_text(json.dumps({"type": "reset", "data": {"seed": 1}}))
            assert ws.receive_json()["type"] == "observation"
            ws.send_text(json.dumps({"type": "close"}))  # what OpenEnv's client sends
        # The server tears the session down on its own loop after the close;
        # wait while that loop still runs (leaving TestClient stops it).
        deadline = time.monotonic() + 5
        while made[-1]._client.closed == 0 and time.monotonic() < deadline:
            time.sleep(0.02)
    assert len(made) == 2 and made[1].calls == 1  # the startup check, then the session
    assert [c._client.closed for c in made] == [1, 1]


def test_focal_actions_are_bounded_and_moves_keep_only_schema_keys():
    with pytest.raises(ValueError):
        ParleySimAction(text="x" * (MAX_ACTION_CHARS + 1))
    with pytest.raises(ValueError, match="limit"):
        ParleySimAction(move={"message": "x" * MAX_ACTION_CHARS, "move": {"type": "none"}})
    env = ParleySimEnvironment(seats=SCRIPTED)
    env.reset(seed=3)
    env.step(ParleySimAction(move={"message": "hi", "move": {"type": "none", "junk": [1] * 100},
                                   "junk": "x" * 1000, "explanation": "wait"}))
    decision = focal_decisions(env)[0]
    assert decision["submitted"] == decision["applied"] == {"message": "hi", "move": {"type": "none"}}
    assert decision["explanation"] == "wait" and decision["strike"] is None


def test_only_allowlisted_routes_are_served_and_none_build_environments():
    made = []
    app = create_parley_app(**PEER, client_factory=recording_factory(made))
    served = {getattr(r, "path", None) for r in app.routes}
    assert served == {"/ws", "/health", "/schema", "/metadata", "/openapi.json", "/docs",
                      "/docs/oauth2-redirect", "/redoc", "/reset", "/step", "/state", "/mcp"}
    client = TestClient(app)
    startup = len(made)
    assert client.get("/metadata").json()["name"] == "ParleySimEnvironment"
    assert client.get("/schema").status_code == 200 and client.get("/health").status_code == 200
    for method, path in [("GET", "/parley_sim/splits"), ("POST", "/parley_sim/tasks"),
                         ("GET", "/list_environments")]:  # OpenEnv's task routes: dropped
        assert client.request(method, path, json={}).status_code == 404, path
    from starlette.websockets import WebSocketDisconnect
    with pytest.raises(WebSocketDisconnect):  # the MCP WebSocket is dropped too
        with client.websocket_connect("/mcp") as ws:
            ws.send_text(json.dumps({"jsonrpc": "2.0", "id": 1, "method": "tools/list"}))
            ws.receive_text()
    assert len(made) == startup  # nothing above built an environment or a provider client


class Rejects:
    """A counterpart client whose provider rejects every request."""

    _client = None

    async def complete(self, **kw):
        err = RuntimeError("Error code: 400 - model 'secret-org/peer-model' is not available")
        err.status_code = 400
        raise err


@pytest.mark.parametrize("mode,leaks", [("never", False), ("after_done", True)])
def test_a_failed_counterpart_turn_names_its_provider_only_when_traces_are_released(mode, leaks):
    env = ParleySimEnvironment(seats={"agent": "client", "customer": "model:peer"},
                               lineup=PEER["lineup"], release_trace=mode,
                               client_factory=lambda spec: Rejects())
    with pytest.raises(Exception) as info:
        env.reset(seed=1)
    assert ("secret-org/peer-model" in str(info.value)) is leaks
    assert env.state.done


def test_schema_projection_keeps_keys_the_schema_allows():
    from parley_sim.adapters.openenv import _project
    closed = {"type": "object", "properties": {"a": {}}, "additionalProperties": False}
    open_ = {"type": "object", "properties": {"a": {}}}
    typed = {"type": "object", "properties": {"a": {}},
             "additionalProperties": {"type": "object", "properties": {"x": {}},
                                      "additionalProperties": False}}
    assert _project({"a": 1, "b": 2}, closed) == {"a": 1}
    assert _project({"a": 1, "b": 2}, open_) == {"a": 1, "b": 2}
    assert _project({"a": 1, "b": {"x": 1, "y": 2}}, typed) == {"a": 1, "b": {"x": 1}}
