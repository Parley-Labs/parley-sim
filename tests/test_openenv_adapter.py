import json

import pytest

pytest.importorskip("openenv")

from fastapi.testclient import TestClient  # noqa: E402
from openenv.core.env_server.serialization import serialize_observation  # noqa: E402

from parley_sim.adapters.openenv import (ParleySimAction, ParleySimEnv,  # noqa: E402
                                         ParleySimEnvironment, create_parley_app)
from parley_sim.envs.support_desk import SupportDesk, make_policy  # noqa: E402
from parley_sim.envs.support_desk.metrics import episode_metrics  # noqa: E402
from parley_sim.runtime import verify  # noqa: E402

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
    reset = client.post("/reset", json={"seed": 5}).json()
    assert reset["done"] is False and reset["observation"]["seat"] == "agent"
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
