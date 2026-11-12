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
        focal_agent=AgentServerRef(type="responses_api_agents", name="focal"),
        counterpart_agents={"peer": AgentServerRef(type="responses_api_agents", name="peer")},
        **cfg)
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
