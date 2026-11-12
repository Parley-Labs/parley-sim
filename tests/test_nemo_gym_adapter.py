import asyncio
import json

import pytest

pytest.importorskip("nemo_gym")

import orjson
from aiohttp import ClientResponseError, RequestInfo
from multidict import CIMultiDict, CIMultiDictProxy
from nemo_gym.base_responses_api_agent import AgentCloseSessionRequest, AgentSeedSessionRequest
from nemo_gym.config_types import AgentServerRef, AggregateMetricsRequest
from nemo_gym.episode_types import EpisodeId, MaterializedTask, TaskId
from nemo_gym.openai_utils import NeMoGymResponse, NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.server_utils import BaseServerConfig, ServerClient
from omegaconf import OmegaConf
from pydantic import BaseModel, ConfigDict
from yarl import URL

from parley_sim.adapters.nemo_gym import (
    ParleySimEnvironmentServer,
    ParleySimEnvironmentServerConfig,
    ParleySimEpisodeRequest,
    ParleySimTaskInput,
)
from parley_sim.envs.support_desk import SupportDesk
from parley_sim.runtime import verify


class _Cookie:
    value = "cookie"


class _Response:
    ok = True
    cookies = {"session": _Cookie()}

    def __init__(self, body: dict) -> None:
        self.body = orjson.dumps(body)

    async def read(self) -> bytes:
        return self.body


def _gym_response(text: str = "", refusal: str | None = None) -> dict:
    part = ({"type": "refusal", "refusal": refusal} if refusal is not None
            else {"type": "output_text", "text": text, "annotations": []})
    return NeMoGymResponse(
        id="r", created_at=0, model="fake-model", object="response",
        output=[{"type": "message", "role": "assistant", "id": "m", "status": "completed",
                 "content": [part]}],
        tool_choice="auto", parallel_tool_calls=True, tools=[],
    ).model_dump(mode="json")


def _turn(move: dict) -> str:
    return json.dumps({"message": "ok", "move": move, "explanation": "test"})


def accept_if_offered(user: str) -> str:
    offered = "Counterpart's latest proposal: {" in user
    return _turn({"type": "accept"} if offered else {"type": "none"})


class _Client(ServerClient):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    calls: list[tuple[str, str]] = []
    requests: list[tuple[str, dict]] = []
    policy: dict = {}  # agent server name -> fn(user_prompt) -> reply text | Exception | ("refusal", why)

    async def post(self, server_name: str, url_path: str, **kwargs) -> _Response:
        body = kwargs.get("json")
        if isinstance(body, BaseModel):
            body = body.model_dump(mode="json", exclude_unset=True)
        self.calls.append((server_name, url_path))
        if url_path == "/v1/agent_sessions":
            seed = AgentSeedSessionRequest.model_validate(body)
            return _Response({"agent_session_id": seed.agent_session_id})
        if url_path == "/v1/agent_sessions/close":
            close = AgentCloseSessionRequest.model_validate(body)
            return _Response({"agent_session_id": close.agent_session_id})
        params = NeMoGymResponseCreateParamsNonStreaming.model_validate(body)
        assert params.text["format"]["type"] == "json_schema"
        self.requests.append((server_name, body))
        user = params.input[-1].content
        reply = self.policy[server_name](user)
        if isinstance(reply, Exception):
            raise reply
        if isinstance(reply, dict):  # a full response body, e.g. one carrying token ids
            return _Response(reply)
        if isinstance(reply, tuple):
            return _Response(_gym_response(refusal=reply[1]))
        return _Response(_gym_response(reply))


def _server(policy: dict, capture_focal: bool = False, **cfg) -> tuple[ParleySimEnvironmentServer, _Client]:
    global_config = OmegaConf.create({
        "focal": {"responses_api_agents": {"focal": {
            "host": "focal", "port": 1, "entrypoint": "app.py", "token_id_capture": capture_focal}}},
        "peer": {"responses_api_agents": {"peer": {
            "host": "peer", "port": 2, "entrypoint": "app.py", "token_id_capture": capture_focal}}},
        "token_id_capture": {"enabled": capture_focal},
    })
    client = _Client(head_server_config=BaseServerConfig(host="head", port=0),
                     global_config_dict=global_config, calls=[], policy=policy)
    config = ParleySimEnvironmentServerConfig(
        name="parley_sim", host="env", port=3, entrypoint="app.py", cleanup_timeout_seconds=10,
        agent_server=AgentServerRef(type="responses_api_agents", name="focal"),
        **{"peer": AgentServerRef(type="responses_api_agents", name="peer"), **cfg})
    return ParleySimEnvironmentServer(config=config, server_client=client), client


def _request(seats: dict, seed: int = 3) -> ParleySimEpisodeRequest:
    return ParleySimEpisodeRequest(
        episode_id=EpisodeId(rollout_id="rollout"),
        task=MaterializedTask(task_id=TaskId(taskset="parley:support_desk", task_id=f"s{seed}"),
                              task_input=ParleySimTaskInput(seed=seed, seats=seats, arm="x")))


def test_scripted_counterpart_episode_runs_and_replays():
    server, client = _server({"focal": accept_if_offered})
    res = asyncio.run(server.run_request(
        _request({"agent": "model:focal", "customer": "policy:conceder"})))
    assert res.failure is None and res.result.status == "complete"
    result = res.result
    assert result.reward == result.metrics["focal_surplus_share"]
    assert result.trace[-1]["final_state"]["end_reason"] == "accepted"
    assert verify(SupportDesk(), result.trace) == []
    paths = [p for _, p in client.calls]
    assert paths[0] == "/v1/agent_sessions" and paths[-1] == "/v1/agent_sessions/close"
    assert paths[1] == "/ng-rollout/rollout/v1/responses"


def test_adaptive_counterpart_routes_seats_and_captures_only_focal():
    peer_turns = iter([_turn({"type": "none"}), _turn({"type": "walk_away"})])
    server, client = _server({"focal": lambda u: _turn({"type": "none"}),
                              "peer": lambda u: next(peer_turns)},
                             capture_focal=True)
    res = asyncio.run(server.run_request(_request({"agent": "model:focal", "customer": "model:peer"})))
    assert res.result.trace[-1]["final_state"]["end_reason"] == "walk_away"
    by_server = {}
    for name, path in client.calls:
        by_server.setdefault(name, []).append(path)
    assert by_server["peer"][1] == "/ng-rollout/rollout/v1/responses"
    assert all("training-token-capture" not in p for p in by_server["peer"])
    assert by_server["focal"][1] == "/ng-rollout/rollout/training-token-capture/v1/responses"
    assert by_server["peer"][-1] == "/v1/agent_sessions/close"
    decisions = [l for l in res.result.trace if l["type"] == "decision"]
    assert [d["seat"] for d in decisions] == ["customer", "agent", "customer"]
    assert decisions[1]["raw"]["attempts"][0]["agent_server"] == "focal"
    assert verify(SupportDesk(), res.result.trace) == []


def test_agent_outage_fails_episode_as_retryable():
    url = URL("http://focal/v1/responses")
    info = RequestInfo(url=url, method="POST", headers=CIMultiDictProxy(CIMultiDict()), real_url=url)
    err = ClientResponseError(request_info=info, history=(), status=503, message="down")
    server, client = _server({"focal": lambda u: err})
    res = asyncio.run(server.run_request(
        _request({"agent": "model:focal", "customer": "policy:hardliner"})))
    assert res.result is None
    assert res.failure.stage == "agent" and res.failure.terminal is False
    assert client.calls[-1][1] == "/v1/agent_sessions/close"


def test_refusals_halt_the_episode():
    server, _ = _server({"focal": lambda u: ("refusal", "nope")}, max_refusals=1)
    res = asyncio.run(server.run_request(
        _request({"agent": "model:focal", "customer": "policy:hardliner"})))
    assert res.failure.stage == "agent" and res.failure.terminal is True
    assert "halted" in res.failure.failure_reason


def test_unknown_counterpart_is_a_terminal_seed_failure():
    server, client = _server({"focal": accept_if_offered})
    res = asyncio.run(server.run_request(_request({"agent": "model:focal", "customer": "model:x"})))
    assert res.failure.stage == "seed" and res.failure.terminal is True
    assert client.calls == []


def test_aggregate_metrics_and_routes():
    server, _ = _server({})
    body = AggregateMetricsRequest(verify_responses=[
        {"reward": 0.5, "metrics": {"deal": 1.0}}, {"reward": 0.0, "metrics": {"deal": 0.0}}])
    agg = asyncio.run(server.aggregate_metrics(body))
    assert agg.key_metrics["mean_reward"] == 0.25 and agg.key_metrics["mean_deal"] == 0.5
    routes = {r.path for r in server.setup_webserver().routes}
    assert {"/run", "/aggregate_metrics"} <= routes


@pytest.mark.parametrize("env,seats,cfg,reason", [
    ("os:getcwd", {"agent": "model:focal", "customer": "policy:conceder"}, {}, "allowed_env_paths"),
    ("support_desk", {"agent": "model:focal", "customer": "policy:conceder"},
     {"reward_metric": "focal_surplus"}, "reward_metric"),
    ("support_desk", {"agent": "model:focal", "customer": "peer"}, {}, "must start with"),
])
def test_unsafe_or_invalid_tasks_fail_at_seed(env, seats, cfg, reason):
    server, client = _server({"focal": accept_if_offered}, **cfg)
    req = ParleySimEpisodeRequest(
        episode_id=EpisodeId(rollout_id="rollout"),
        task=MaterializedTask(task_id=TaskId(taskset="parley:support_desk", task_id="s3"),
                              task_input=ParleySimTaskInput(seed=3, seats=seats, env=env)))
    res = asyncio.run(server.run_request(req))
    assert res.failure is not None and res.failure.stage == "seed"
    assert reason in res.failure.failure_reason and client.calls == []


def test_registered_env_names_with_a_colon_are_allowed():
    from parley_sim import envs
    envs.register("vendor:desk", envs.get("support_desk"))
    server, _ = _server({"focal": accept_if_offered})
    req = ParleySimEpisodeRequest(
        episode_id=EpisodeId(rollout_id="rollout"),
        task=MaterializedTask(task_id=TaskId(taskset="parley:support_desk", task_id="s3"),
                              task_input=ParleySimTaskInput(
                                  seed=3, seats={"agent": "model:focal", "customer": "policy:conceder"},
                                  env="vendor:desk")))
    res = asyncio.run(server.run_request(req))
    assert res.failure is None and res.result.status == "complete"


def _token_response(text: str, prompt: list[int], generated: list[int]) -> dict:
    body = _gym_response(text)
    body["output"][0].update(prompt_token_ids=prompt, generation_token_ids=generated,
                             generation_log_probs=[-0.1] * len(generated))
    return NeMoGymResponse.model_validate(body).model_dump(mode="json")


def test_focal_tokens_reach_the_collected_rollout_and_the_conversation_is_continuous():
    from nemo_gym.rollout_collection import _episode_record
    from nemo_gym.token_id_capture.delivery import rollout_carries_token_ids

    focal_turns = iter(range(10))
    peer_moves = iter([{"type": "none"}, {"type": "none"}, {"type": "walk_away"}])

    def focal(user):
        k = next(focal_turns)
        return _token_response(_turn({"type": "none"}), prompt=[1, 2, k], generated=[100 + k])

    def peer(user):
        return _token_response(_turn(next(peer_moves)), prompt=[7], generated=[900])

    server, client = _server({"focal": focal, "peer": peer})
    res = asyncio.run(server.run_request(_request({"agent": "model:focal", "customer": "model:peer"})))
    assert res.failure is None, res.failure
    result = res.result
    assert len(result.focal_calls) >= 2

    # The training payload holds every focal reply's tokens, in order, and no counterpart tokens.
    generated = [item["generation_token_ids"] for item in result.response["output"]]
    assert generated == [[100 + k] for k in range(len(result.focal_calls))]

    # The training payload is a valid response under the pinned NeMo schema.
    NeMoGymResponse.model_validate(result.response)

    # Gym's collector turns the episode reply into the rollout record a trainer reads.
    record = _episode_record(res.model_dump(mode="json"))
    assert rollout_carries_token_ids(record)
    assert record["response"]["output"] == result.response["output"]

    # Each focal prompt extends the previous one: earlier input, then the agent's own
    # output items verbatim, then the new user message. That keeps tokens continuous.
    focal_inputs = [body["input"] for name, body in client.requests if name == "focal"]
    for k in range(1, len(focal_inputs)):
        before, after = focal_inputs[k - 1], focal_inputs[k]
        assert after[:len(before)] == before
        assert after[len(before):-1] == result.focal_calls[k - 1]["response"]["output"]
        assert after[-1]["role"] == "user"


def test_focal_conversation_can_be_turned_off():
    server, client = _server({"focal": accept_if_offered}, focal_conversation=False)
    res = asyncio.run(server.run_request(
        _request({"agent": "model:focal", "customer": "policy:conceder"})))
    assert res.result.response is None and res.result.focal_calls == []
    assert all(len(body["input"]) == 2 for name, body in client.requests if name == "focal")


def test_a_refusal_stays_out_of_the_conversation_and_the_agent_keeps_playing():
    replies = iter([_turn({"type": "none"}), ("refusal", "temporary refusal"),
                    _turn({"type": "walk_away"})])
    server, client = _server({"focal": lambda u: next(replies)}, max_refusals=3)
    res = asyncio.run(server.run_request(
        _request({"agent": "model:focal", "customer": "policy:hardliner"})))
    assert res.failure is None
    focal = [d for d in res.result.trace if d["type"] == "decision" and d["seat"] == "agent"]
    assert [d["strike"] for d in focal] == [None, "refusal", None]
    assert focal[-1]["applied"]["move"] == {"type": "walk_away"}
    inputs = [body["input"] for name, body in client.requests if name == "focal"]
    assert len(inputs) == 3
    # The third prompt continues from the first reply, skipping the refused call,
    # and its update covers both customer moves since that reply.
    assert inputs[2][:len(inputs[0])] == inputs[0]
    assert inputs[2][len(inputs[0]):-1] == res.result.focal_calls[0]["response"]["output"]
    assert inputs[2][-1]["content"].count("[Counterpart]") == 2
    assert [c["refused"] for c in res.result.focal_calls] == [False, True, False]
    assert len(res.result.response["output"]) == 2  # the refused call is not training data


def test_update_says_when_the_agents_move_was_not_applied():
    from parley_sim.envs.support_desk import scenario as sc
    inst = SupportDesk().generate(0, {})
    limits = sc.instance_to_json(inst)["agent"]["limits"]
    over = {"discount_pct": 40, "credit_usd": 100, "upgrade": "none", "term": "monthly"}
    assert over["discount_pct"] > limits["discount_pct"] or over["credit_usd"] > limits["credit_usd"]
    replies = iter([json.dumps({"message": "I offer you this package.",
                                "move": {"type": "propose", "deal": over}, "explanation": "x"}),
                    _turn({"type": "walk_away"})])
    server, client = _server({"focal": lambda u: next(replies)})
    res = asyncio.run(server.run_request(
        _request({"agent": "model:focal", "customer": "policy:hardliner"}, seed=0)))
    focal = [d for d in res.result.trace if d["type"] == "decision" and d["seat"] == "agent"]
    assert focal[0]["strike"] == "illegal_action"
    inputs = [body["input"] for name, body in client.requests if name == "focal"]
    # The sampled reply stays in the conversation for continuity...
    assert "I offer you this package." in json.dumps(inputs[1][:-1])
    # ...and the next update says what was actually recorded, and that no own offer stands.
    update = inputs[1][-1]["content"]
    assert "could not be applied as sent" in update and '(move: none)' in update
    assert "Your standing proposal: none" in update


# --- Review fixes -------------------------------------------------------------------------


def _sparse(text: str, k: int = 0, **extra) -> dict:
    """A response body as a model server sends it: only the fields it sets."""
    return {"id": f"resp-{k}", "created_at": 0, "model": "fake-model", "object": "response",
            "output": [{"type": "message", "role": "assistant", "id": f"m{k}", "status": "completed",
                        "content": [{"type": "output_text", "text": text, "annotations": []}],
                        "prompt_token_ids": [1, 2, k], "generation_token_ids": [100 + k],
                        "generation_log_probs": [-0.1]}],
            "tool_choice": "auto", "parallel_tool_calls": True, "tools": [],
            "usage": {"input_tokens": 10 + k, "output_tokens": 1, "total_tokens": 11 + k,
                      "input_tokens_details": {"cached_tokens": 0},
                      "output_tokens_details": {"reasoning_tokens": 0}},
            **extra}


def _focal_decisions(result) -> list[dict]:
    return [d for d in result.trace if d["type"] == "decision" and d["seat"] == "agent"]


def _docs_block(lang: str) -> str:
    from pathlib import Path
    text = Path(__file__).parents[1].joinpath("docs/adapters.md").read_text()
    section = text[text.index("## NeMo Gym"):text.index("## OpenEnv")]
    return section.split(f"```{lang}\n", 1)[1].split("```", 1)[0]


def test_an_echoing_agent_is_read_and_its_prompt_echo_ignored():
    # Gym's output items admit prompt messages with plain-string content (local agents
    # echo their trajectory). They used to crash the reader: every turn became an
    # `exception` strike and the episode completed on fallback moves.
    turns = iter(range(10))

    def focal(user):
        k = next(turns)
        body = _sparse(_turn({"type": "none"} if k < 2 else {"type": "walk_away"}), k)
        body["output"].insert(0, {"type": "message", "role": "user", "content": user[:20]})
        return body

    server, client = _server({"focal": focal})
    res = asyncio.run(server.run_request(
        _request({"agent": "model:focal", "customer": "policy:hardliner"})))
    assert res.failure is None, res.failure
    focal_turns = _focal_decisions(res.result)
    assert [d["strike"] for d in focal_turns] == [None, None, None]
    assert focal_turns[-1]["applied"]["move"] == {"type": "walk_away"}
    assert len(res.result.focal_calls) == sum(1 for name, _ in client.requests if name == "focal")


def test_an_unprocessable_reply_aborts_the_episode_instead_of_scoring_a_strike(monkeypatch):
    from parley_sim.adapters import nemo_gym

    def broken(response):
        raise AttributeError("adapter bug")

    monkeypatch.setattr(nemo_gym, "_read_output", broken)
    server, client = _server({"focal": accept_if_offered})
    res = asyncio.run(server.run_request(
        _request({"agent": "model:focal", "customer": "policy:hardliner"})))
    assert res.result is None
    assert res.failure.stage == "agent" and res.failure.terminal is True
    assert "could not process the agent's reply" in res.failure.failure_reason
    assert "adapter bug" in res.failure.failure_reason
    assert client.calls[-1][1] == "/v1/agent_sessions/close"


def test_the_documented_task_row_is_routed_and_run_the_way_gyms_collector_does():
    from types import SimpleNamespace

    from fastapi.testclient import TestClient
    from nemo_gym.rollout_collection import (
        _environment_server_for_config_row,
        _episode_request_body,
        _materialized_taskset,
    )

    doc_row = json.loads(_docs_block("json"))
    routes = OmegaConf.to_container(OmegaConf.create(_docs_block("yaml")))["environment_server_routes"]
    row = {**doc_row, "_ng_task_index": 0, "_ng_rollout_index": 0}
    # The collector routes a materialized row by its taskset, in every routing mode...
    assert _materialized_taskset(row) == "parley_sim"
    for mode in ("agent", "legacy", "taskset"):
        config = SimpleNamespace(environment_server_routes=routes, environment_routing_mode=mode,
                                 environment_server_name="parley_sim")
        assert _environment_server_for_config_row(row, config) == "parley_sim"
    # ...and posts it to /run as an episode request, which the server runs.
    server, _ = _server({"focal": accept_if_offered})
    http = TestClient(server.setup_webserver())
    reply = http.post("/run", json=_episode_request_body(row))
    assert reply.status_code == 200 and reply.json()["failure"] is None
    assert reply.json()["result"]["trace"][0]["spec"]["pair_key"] == "run-1000-focal"
    # A flat row is no episode request: the collector never routes it here by taskset,
    # and posted as-is it is rejected.
    flat = doc_row["task_input"]
    assert _materialized_taskset(flat) is None
    assert http.post("/run", json=flat).status_code == 422


def _gym_config(peer_name: str = "peer", define_peer: bool = True):
    import nemo_gym.global_config as gc

    config = OmegaConf.create(_docs_block("yaml"))
    env = config.parley_sim.environment_servers.parley_sim
    env.agent_server.name, env.peer.name = "focal", peer_name
    agent = {"responses_api_agents": {"simple_agent": {
        "entrypoint": "app.py", "model_server": {"type": "responses_api_models", "name": "policy"}}}}
    config.focal = agent
    if define_peer:
        config.peer = agent
    config.policy = {"responses_api_models": {"openai_model": {"entrypoint": "app.py"}}}
    config[gc.ERROR_ON_AGENT_WITHOUT_ENVIRONMENT_SERVER_KEY_NAME] = True
    return gc.GlobalConfigDictParserConfig(initial_global_config_dict=config, skip_load_from_cli=True,
                                           skip_load_from_dotenv=True, offline=True)


def test_the_documented_config_loads_in_gyms_strict_mode_with_focal_attribution(caplog):
    import warnings

    import nemo_gym.global_config as gc
    from nemo_gym.config_types import ServerRefNotFoundError
    from omegaconf import DictConfig

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        parsed = gc.GlobalConfigDictParser().parse(_gym_config())
    # No legacy relay generated for the counterpart, and no deprecation warning.
    servers = [k for k, v in parsed.items() if isinstance(v, DictConfig) and "environment_servers" in v]
    assert servers == ["parley_sim"]
    assert not [r for r in caplog.records if "legacy" in r.getMessage().lower()]
    env = parsed.parley_sim.environment_servers.parley_sim
    assert gc.environment_server_agent_names(env) == ["focal", "peer"]
    # Results (and so token-capture finalisation) are attributed to the focal agent.
    assert gc.environment_server_attributed_agent(env) == "focal"
    # The server builds its config from the same block.
    config = ParleySimEnvironmentServerConfig.model_validate(
        {**OmegaConf.to_container(env), "name": "parley_sim"})
    assert config.agent_server.name == "focal"
    assert {k: v.name for k, v in config.counterpart_agents.items()} == {"peer": "peer"}
    # Counterparts are existence-checked like any agent reference.
    with pytest.raises(ServerRefNotFoundError, match="field 'peer'"):
        gc.GlobalConfigDictParser().parse(_gym_config("missing", define_peer=False))


@pytest.mark.parametrize("extra,message", [
    ({"focal_agent": {"type": "responses_api_agents", "name": "focal"}}, "now agent_server"),
    ({"counterpart_agents": {"peer": {"type": "responses_api_agents", "name": "p"}}}, "top-level"),
    ({"reward_metirc": "deal"}, "unknown config key 'reward_metirc'"),
])
def test_config_keys_beyond_the_declared_ones_must_be_counterpart_agents(extra, message):
    from pydantic import ValidationError

    with pytest.raises(ValidationError, match=message):
        ParleySimEnvironmentServerConfig(
            name="parley_sim", host="env", port=3, entrypoint="app.py", cleanup_timeout_seconds=10,
            agent_server={"type": "responses_api_agents", "name": "focal"}, **extra)


@pytest.mark.parametrize("env,focal_seat,allowed,reason", [
    ("support_desk", "bogus", [], "focal_seat 'bogus'"),
    ("os:getcwd", None, ["os:getcwd"], "TypeError"),
    ("parley_sim_no_such_module:ENTRY", None, ["parley_sim_no_such_module:ENTRY"],
     "ModuleNotFoundError"),
    ("os:NO_SUCH_ATTR", None, ["os:NO_SUCH_ATTR"], "has no attribute"),
])
def test_bad_focal_seats_and_unloadable_env_paths_fail_at_seed(env, focal_seat, allowed, reason):
    server, client = _server({"focal": accept_if_offered}, allowed_env_paths=allowed)
    req = ParleySimEpisodeRequest(
        episode_id=EpisodeId(rollout_id="rollout"),
        task=MaterializedTask(task_id=TaskId(taskset="parley_sim", task_id="s3"),
                              task_input=ParleySimTaskInput(
                                  seed=3, seats={"agent": "model:focal", "customer": "policy:conceder"},
                                  env=env, focal_seat=focal_seat)))
    res = asyncio.run(server.run_request(req))
    assert res.failure is not None and res.failure.stage == "seed" and res.failure.terminal
    assert reason in res.failure.failure_reason and client.calls == []


def test_an_undefined_reward_fails_the_episode_instead_of_scoring_zero():
    server, _ = _server({"focal": lambda u: _turn({"type": "walk_away"})},
                        reward_metric="cp_first_offer_share")
    req = ParleySimEpisodeRequest(
        episode_id=EpisodeId(rollout_id="rollout"),
        task=MaterializedTask(task_id=TaskId(taskset="parley_sim", task_id="s3"),
                              task_input=ParleySimTaskInput(
                                  seed=3, seats={"agent": "policy:agent_scripted",
                                                 "customer": "model:focal"},
                                  focal_seat="customer")))
    res = asyncio.run(server.run_request(req))
    assert res.result is None
    assert res.failure.stage == "verification" and res.failure.terminal is True
    assert res.failure.failure_kind == "parley_sim:reward_undefined"
    assert "cp_first_offer_share" in res.failure.failure_reason


def test_replayed_output_items_are_exactly_what_the_agent_returned_minus_gym_keys():
    turns = iter(range(10))
    served = []

    def focal(user):
        k = next(turns)
        body = _sparse(_turn({"type": "none"} if k < 2 else {"type": "walk_away"}), k,
                       _ng_trajectory={"task_id": "t", "invocations": []})
        served.append(json.loads(json.dumps(body)))
        return body

    server, client = _server({"focal": focal})
    res = asyncio.run(server.run_request(
        _request({"agent": "model:focal", "customer": "policy:hardliner"})))
    assert res.failure is None, res.failure
    inputs = [body["input"] for name, body in client.requests if name == "focal"]
    assert len(inputs) == len(served) == 3
    for k in (1, 2):
        # Exactly the items served (no unset optionals filled in as nulls).
        assert inputs[k][len(inputs[k - 1]):-1] == served[k - 1]["output"]
    result = res.result
    assert result.response["output"] == [item for body in served for item in body["output"]]
    assert all(not k.startswith("_ng_") for k in result.response)
    assert all(not k.startswith("_ng_") for c in result.focal_calls for k in c["response"])
    NeMoGymResponse.model_validate(result.response)
    # The envelope is the last kept call's, with its served id: Gym's token-capture
    # finaliser uses response["id"] to find the terminal captured call.
    assert result.response["id"] == "resp-2"


def test_conversation_off_keeps_focal_calls_off_the_capture_route():
    server, client = _server({"focal": accept_if_offered}, capture_focal=True,
                             focal_conversation=False)
    res = asyncio.run(server.run_request(
        _request({"agent": "model:focal", "customer": "policy:conceder"})))
    assert res.failure is None and res.result.response is None
    paths = [p for name, p in client.calls if name == "focal" and p.endswith("/v1/responses")]
    assert paths and all("training-token-capture" not in p for p in paths)


class _CleanupStub:
    def register_cleanup(self, name, callback):
        return None


def test_an_empty_reply_is_not_an_exchange_and_the_seat_retry_stays_in_step():
    """Simulates ModelSeat's conversation mode: an empty reply is not appended to the
    seat's history, and the retry goes out with the same history and the correction
    appended to the original prompt."""
    from parley_sim.adapters.nemo_gym import _AgentSessionClient

    replies = iter([_sparse(_turn({"type": "none"}), 0), _sparse("  \n", 1),
                    _sparse(_turn({"type": "none"}), 2)])
    server, client = _server({"focal": lambda u: next(replies)})
    req = _request({"agent": "model:focal", "customer": "policy:hardliner"})
    session = _AgentSessionClient(server, req, _CleanupStub(), server.config.agent_server, trained=True)
    history: list[dict] = []

    async def turn(user):
        for attempt in range(2):
            prompt = user if attempt == 0 else f"{user}\n\nYour previous reply was invalid."
            text, _ = await session.complete(system="sys", user=prompt, schema={"type": "object"},
                                               history=list(history))
            if text.strip():
                history.append({"user": prompt, "assistant": text})
                return text
        raise AssertionError("no reply")

    async def play():
        await turn("turn 1")
        await turn("turn 2")  # empty, then the retry

    asyncio.run(play())
    inputs = [body["input"] for name, body in client.requests if name == "focal"]
    first_output = session.calls[0]["response"]["output"]
    # The empty reply's request and its retry carry the same conversation so far.
    assert inputs[1][:-1] == inputs[2][:-1] == [inputs[0][0], inputs[0][1], *first_output]
    assert inputs[2][-1]["content"].startswith(inputs[1][-1]["content"])
    assert [u for u, _ in session.exchanges] == ["turn 1", "turn 2\n\nYour previous reply was invalid."]
    assert [(c["empty"], c["in_conversation"]) for c in session.calls] == [
        (False, True), (True, False), (False, True)]
    payload = session.training_response()
    assert [i["generation_token_ids"] for i in payload["output"]] == [[100], [102]]
    assert payload["id"] == "resp-2"
    # A third turn sent with the seat's history (2 exchanges) is in step, not an abort.
    assert len(history) == len(session.exchanges)


def test_empty_replies_through_the_seat_are_retried_and_kept_out_of_training_data():
    replies = iter([_sparse(_turn({"type": "none"}), 0), _sparse("", 1),
                    _sparse(_turn({"type": "none"}), 2), _sparse(_turn({"type": "walk_away"}), 3)])
    server, _ = _server({"focal": lambda u: next(replies)})
    res = asyncio.run(server.run_request(
        _request({"agent": "model:focal", "customer": "policy:hardliner"})))
    assert res.failure is None, res.failure
    assert [d["strike"] for d in _focal_decisions(res.result)] == [None, None, None]
    calls = res.result.focal_calls
    assert [c["empty"] for c in calls] == [False, True, False, False]
    assert [c["in_conversation"] for c in calls] == [True, False, True, True]
    assert [i["generation_token_ids"] for i in res.result.response["output"]] == [[100], [102], [103]]


def test_a_reply_cut_off_at_the_output_cap_is_a_truncated_strike():
    replies = iter([_sparse('{"message": "cut', 0, status="incomplete",
                            incomplete_details={"reason": "max_output_tokens"}),
                    _sparse(_turn({"type": "walk_away"}), 1)])
    server, _ = _server({"focal": lambda u: next(replies)})
    res = asyncio.run(server.run_request(
        _request({"agent": "model:focal", "customer": "policy:hardliner"})))
    assert res.failure is None, res.failure
    first = _focal_decisions(res.result)[0]
    assert first["strike"] == "truncated"
    assert first["raw"]["attempts"][0]["stop_reason"] == "max_tokens"


def test_a_refused_call_keeps_its_usage():
    replies = iter([("refusal", "nope"), _turn({"type": "walk_away"})])
    server, _ = _server({"focal": lambda u: next(replies)}, max_refusals=3)
    res = asyncio.run(server.run_request(
        _request({"agent": "model:focal", "customer": "policy:hardliner"})))
    assert res.failure is None, res.failure
    refused = _focal_decisions(res.result)[0]
    assert refused["strike"] == "refusal"
    attempt = refused["raw"]["attempts"][0]
    assert attempt["agent_server"] == "focal" and attempt["stop_reason"] == "refusal"
    assert "usage" in attempt and res.result.focal_calls[0]["refused"] is True


# --- Over real HTTP: Gym's SimpleAgent in front of a fake model server ---------------------


def _free_port() -> int:
    import socket
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


async def _http_episode(model_reply, *, observability: bool = False):
    """Run one episode through a real Gym SimpleAgent over HTTP. `model_reply(n)` is the
    fake model server's n-th response body. Returns (episode response, model requests)."""
    import aiohttp
    import nemo_gym.server_utils as su
    import uvicorn
    from fastapi import FastAPI, Request
    from nemo_gym.config_types import ModelServerRef
    from responses_api_agents.simple_agent.app import SimpleAgent, SimpleAgentConfig

    agent_port, model_port = _free_port(), _free_port()
    global_config = OmegaConf.create({
        "focal": {"responses_api_agents": {"simple_agent": {
            "host": "127.0.0.1", "port": agent_port, "entrypoint": "app.py"}}},
        "model": {"responses_api_models": {"fake": {
            "host": "127.0.0.1", "port": model_port, "entrypoint": "app.py"}}},
        "token_id_capture": {"enabled": False}, "observability_enabled": observability})
    client = ServerClient(head_server_config=BaseServerConfig(host="127.0.0.1", port=1),
                          global_config_dict=global_config)
    requests: list[dict] = []
    model = FastAPI()

    async def handle(request: Request):
        requests.append(await request.json())
        return model_reply(len(requests) - 1)

    model.post("/v1/responses")(handle)
    model.post("/ng-rollout/{rid}/v1/responses")(handle)
    agent = SimpleAgent(config=SimpleAgentConfig(
        name="focal", host="127.0.0.1", port=agent_port, entrypoint="app.py",
        model_server=ModelServerRef(type="responses_api_models", name="model")), server_client=client)
    servers = []
    for app, port in ((model, model_port), (agent.setup_webserver(), agent_port)):
        server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error"))
        server.install_signal_handlers = lambda: None
        servers.append((server, asyncio.create_task(server.serve())))
    previous = su._GLOBAL_AIOHTTP_CLIENT
    su._GLOBAL_AIOHTTP_CLIENT = aiohttp.ClientSession(cookie_jar=aiohttp.DummyCookieJar())
    try:
        while not all(s.started for s, _ in servers):
            await asyncio.sleep(0.02)
        env = ParleySimEnvironmentServer(config=ParleySimEnvironmentServerConfig(
            name="parley_sim", host="127.0.0.1", port=3, entrypoint="app.py", cleanup_timeout_seconds=10,
            agent_server=AgentServerRef(type="responses_api_agents", name="focal")), server_client=client)
        res = await env.run_request(_request({"agent": "model:focal", "customer": "policy:hardliner"}))
    finally:
        await su._GLOBAL_AIOHTTP_CLIENT.close()
        su._GLOBAL_AIOHTTP_CLIENT = previous
        for server, _ in servers:
            server.should_exit = True
        await asyncio.gather(*[task for _, task in servers], return_exceptions=True)
    return res, requests


def _drop_nulls(value):
    if isinstance(value, dict):
        return {k: _drop_nulls(v) for k, v in value.items() if v is not None}
    if isinstance(value, list):
        return [_drop_nulls(v) for v in value]
    return value


def test_over_http_the_model_gets_its_own_items_back_and_no_gym_keys_leak():
    moves = [{"type": "none"}, {"type": "none"}, {"type": "walk_away"}]

    def reply(n):
        reasoning = {"type": "reasoning", "id": f"rs{n}",
                     "summary": [{"type": "summary_text", "text": "think"}],
                     "prompt_token_ids": [1, 2, n], "generation_token_ids": [50 + n],
                     "generation_log_probs": [-0.2]}
        body = _sparse(_turn(moves[min(n, 2)]), n)
        body["output"].insert(0, reasoning)
        return body

    res, requests = asyncio.run(_http_episode(reply, observability=True))
    assert res.failure is None, res.failure
    assert len(requests) == 3
    result = res.result
    for k in (1, 2):
        # The resent items are exactly what the agent returned. SimpleAgent's own HTTP
        # reply spells out unset optionals as nulls; beyond those, they are the model
        # server's items unchanged.
        resent = requests[k]["input"][len(requests[k - 1]["input"]):-1]
        assert resent == result.focal_calls[k - 1]["response"]["output"]
        assert _drop_nulls(resent) == reply(k - 1)["output"]
    # With observability on, the agent attaches `_ng_trajectory`; it stays out of the payload.
    assert all(not k.startswith("_ng_") for k in result.response)
    assert all(not k.startswith("_ng_") for c in result.focal_calls for k in c["response"])
    assert [i["generation_token_ids"] for i in result.response["output"]] == [
        [50], [100], [51], [101], [52], [102]]
    assert result.response["id"] == "resp-2"
