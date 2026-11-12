"""NeMo Gym environment server for parley-sim episodes (`pip install parley-sim[gym]`).

One `/run` is one parley-sim episode: the Episode runtime advances scripted
seats in-process, and every model seat's turn goes to a Gym agent server
through an agent session (`/v1/agent_sessions`, then
`/ng-rollout/<capture_key>/v1/responses`). The focal seat is the agent under
training or evaluation, and only its calls use the token-capture route. The
episode's reward is the env's focal metric (default `focal_surplus_share`);
the full trace, per-seat outcomes and all env metrics ride in the result.

Targets the nemo-gym 0.7 API. To serve it from a Gym checkout, add
`environment_servers/parley_sim/app.py`:

    from nemo_gym.server_utils import is_nemo_gym_fastapi_entrypoint
    from parley_sim.adapters.nemo_gym import ParleySimEnvironmentServer

    if __name__ == "__main__":
        ParleySimEnvironmentServer.run_webserver()
    elif is_nemo_gym_fastapi_entrypoint(__file__):
        app = ParleySimEnvironmentServer.run_webserver()

and `environment_servers/parley_sim/configs/parley_sim.yaml`:

    parley_sim:
      environment_servers:
        parley_sim:
          entrypoint: app.py
          focal_agent: {type: responses_api_agents, name: ???}
          counterpart_agents:            # lineup names used as "model:<name>" seats
            peer: {type: responses_api_agents, name: ???}
          cleanup_timeout_seconds: 180
          default_episode_timeout_seconds: 3600

Task rows carry a `ParleySimTaskInput`, e.g.
`{"seed": 1000, "seats": {"agent": "model:focal", "customer": "policy:hardliner"},
"arm": "soc", "pair_key": "run-1000-focal"}`.
"""
from __future__ import annotations

import asyncio
import statistics
from typing import Any
from uuid import uuid4

from aiohttp import ClientConnectionError, ClientPayloadError, ClientResponseError
from fastapi import Body
from nemo_gym.base_environment_server import (
    BaseEnvironmentServer,
    BaseEnvironmentServerConfig,
    CleanupContext,
    HandledEpisodeError,
)
from nemo_gym.base_responses_api_agent import (
    AgentCloseSessionRequest,
    AgentSeedSessionRequest,
    AgentSeedSessionResponse,
)
from nemo_gym.config_types import (
    TOKEN_CAPTURE_PATH_SEGMENT,
    AgentServerRef,
    AggregateMetrics,
    AggregateMetricsRequest,
)
from nemo_gym.episode_types import BaseEpisodeRequest, BaseEpisodeResponse, EpisodeFailure
from nemo_gym.global_config import TOKEN_ID_CAPTURE_BLOCK, get_first_server_config_dict
from nemo_gym.openai_utils import NeMoGymResponse, NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.server_utils import get_response_json, raise_for_status
from pydantic import BaseModel, ConfigDict, Field

from parley_sim import envs
from parley_sim.core.protocol import JSON, Seat
from parley_sim.drivers.local import scripted_policies
from parley_sim.drivers.models import (
    Breaker,
    EpisodeAbort,
    ModelSeat,
    ModelSpec,
    ProviderHalted,
    ProviderRefusal,
)
from parley_sim.runtime.episode import Episode, EpisodeComplete, EpisodeSpec


class ParleySimTaskInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    seed: int
    seats: dict[Seat, str]  # seat -> "policy:<name>" | "model:<lineup name>"
    env: str = "support_desk"
    env_config: dict[str, Any] = Field(default_factory=dict)
    focal_seat: Seat | None = None
    arm: str | None = None
    pair_key: str | None = None
    cell_key: str | None = None
    replicate: int = 0


class ParleySimEpisodeResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    reward: float
    status: str
    outcomes: dict[Seat, float]
    metrics: dict[str, float | None]
    n_decisions: int
    n_strikes: int
    trace: list[dict[str, Any]]
    # The focal seat's training payload: a Gym-native Responses object whose
    # output items are every focal reply in order, token ids and log probs
    # included as the agent returned them. Counterpart replies are never in
    # it. Gym's collector reads it as the rollout's `response`.
    response: dict[str, Any] | None = None
    # Every focal model call in full (request and response), for audit.
    focal_calls: list[dict[str, Any]] = Field(default_factory=list)


class ParleySimEpisodeRequest(BaseEpisodeRequest[ParleySimTaskInput]):
    """Request for the parley-sim episode protocol."""


class ParleySimEpisodeResponse(BaseEpisodeResponse[ParleySimEpisodeResult]):
    """Response for the parley-sim episode protocol."""


class ParleySimEnvironmentServerConfig(BaseEnvironmentServerConfig):
    model_config = ConfigDict(extra="forbid")

    focal_agent: AgentServerRef
    counterpart_agents: dict[str, AgentServerRef] = Field(default_factory=dict)
    reward_metric: str = "focal_surplus_share"
    actor_call_timeout_seconds: float = Field(300.0, gt=0)
    max_refusals: int = Field(3, ge=1)
    # Play the focal seat as one growing conversation, so its token sequence
    # is continuous across turns and the episode yields one training sample.
    focal_conversation: bool = True
    # Environments task rows may name by import path ("pkg.module:ENTRY").
    # Names of built-in, registered or entry-point environments are always
    # allowed; an import path in a task row is refused unless listed here,
    # because loading one imports and runs code named by the task data.
    allowed_env_paths: list[str] = Field(default_factory=list)
    # Merged into every Responses API request (e.g. {"reasoning": {"effort": "low"}}).
    responses_create_params: dict[str, Any] = Field(default_factory=dict)


class _AgentSessionClient:
    """parley-sim ProviderClient that answers through one Gym agent session."""

    def __init__(self, server: ParleySimEnvironmentServer, request: ParleySimEpisodeRequest,
                 cleanup: CleanupContext, target: AgentServerRef, trained: bool):
        self.server, self.request, self.cleanup = server, request, cleanup
        self.target, self.trained = target, trained
        self.session_id: str | None = None
        self.cookies: dict[str, str] = {}
        # Conversation seats: (user message, output items) per completed call.
        self.exchanges: list[tuple[str, list[dict[str, Any]]]] = []
        self.calls: list[dict[str, Any]] = []

    async def _seed(self) -> None:
        session_id = f"agent-session-{uuid4().hex}"
        self.session_id = session_id

        async def close() -> None:
            response = await self.server.server_client.post(
                server_name=self.target.name, url_path="/v1/agent_sessions/close",
                json=AgentCloseSessionRequest(agent_session_id=session_id,
                                              episode_id=self.request.episode_id
                                              ).model_dump(mode="json"),
                cookies=self.cookies)
            await raise_for_status(response)

        self.cleanup.register_cleanup(f"agent session {self.target.name}", close)
        response = await self.server.server_client.post(
            server_name=self.target.name, url_path="/v1/agent_sessions",
            json=AgentSeedSessionRequest(agent_session_id=session_id,
                                         episode_id=self.request.episode_id,
                                         task_id=self.request.task.task_id,
                                         tool_accesses=[]).model_dump(mode="json"))
        await raise_for_status(response)
        seeded = AgentSeedSessionResponse.model_validate(await get_response_json(response))
        if seeded.agent_session_id != session_id:
            raise ValueError("Agent seed returned a different agent_session_id")
        self.cookies = _cookies(response)

    def _input(self, system: str, user: str, history: list[JSON] | None) -> list[Any]:
        """The request input. A conversation seat resends its earlier turns with
        the agent's own output items verbatim, so each prompt extends the
        previous one token for token."""
        if history is None:
            return [{"role": "system", "content": system}, {"role": "user", "content": user}]
        if len(history) != len(self.exchanges):
            # A bug, not a seat failure: end the episode rather than score fallbacks.
            raise EpisodeAbort("conversation history is out of step with the agent session",
                               retryable=False)
        items: list[Any] = [{"role": "system", "content": system}]
        for prior_user, output in self.exchanges:
            items += [{"role": "user", "content": prior_user}, *output]
        return [*items, {"role": "user", "content": user}]

    def training_response(self) -> dict[str, Any] | None:
        """Every reply of this conversation seat as one valid NeMoGymResponse:
        the last reply's envelope, the conversation's output items in order,
        and usage covering the whole conversation (input = the final prompt,
        which contains every earlier turn; output = all generated tokens)."""
        kept = [c["response"] for c in self.calls if not c.get("refused")]
        if not kept:
            return None
        usage = [r.get("usage") or {} for r in kept]
        n_in = usage[-1].get("input_tokens", 0) or 0
        n_out = sum(u.get("output_tokens", 0) or 0 for u in usage)
        envelope = {**kept[-1],
                    "id": f"parley-{self.request.episode_id.rollout_id}-{self.target.name}",
                    "output": [item for _, output in self.exchanges for item in output],
                    "usage": {"input_tokens": n_in, "output_tokens": n_out,
                              "total_tokens": n_in + n_out,
                              "input_tokens_details": {"cached_tokens": 0},
                              "output_tokens_details": {"reasoning_tokens": sum(
                                  (u.get("output_tokens_details") or {}).get("reasoning_tokens", 0)
                                  or 0 for u in usage)}}}
        return NeMoGymResponse.model_validate(envelope).model_dump(mode="json")

    async def complete(self, *, system: str, user: str, schema: JSON,
                       history: list[JSON] | None = None) -> tuple[Any, JSON]:
        params = {
            **self.server.config.responses_create_params,
            "input": self._input(system, user, history),
            "text": {"format": {"type": "json_schema", "name": "turn", "schema": schema}},
        }
        try:
            async with asyncio.timeout(self.server.config.actor_call_timeout_seconds):
                if self.session_id is None:
                    await self._seed()
                response = await self.server.server_client.post(
                    server_name=self.target.name,
                    url_path=self.server.responses_path(self.target.name, self.request,
                                                        self.trained),
                    json=NeMoGymResponseCreateParamsNonStreaming.model_validate(params),
                    cookies=self.cookies)
                await raise_for_status(response)
                self.cookies = _cookies(response) or self.cookies
                result = NeMoGymResponse.model_validate(await get_response_json(response))
        except Exception as error:
            raise EpisodeAbort(f"agent {self.target.name}: {_describe(error)}"[:2000],
                               retryable=_is_retryable(error)) from error
        dumped = result.model_dump(mode="json")
        text, refusal, reasoning = _read_output(result)
        if history is not None:
            self.calls.append({"request": params, "response": dumped,
                               "refused": refusal is not None and not text})
        if refusal is not None and not text:
            # Not part of the conversation: the seat records no exchange for it
            # either, so both sides stay in step and the next turn resumes from
            # the last reply the agent actually gave.
            raise ProviderRefusal(refusal or "refusal")
        if history is not None:
            self.exchanges.append((user, dumped["output"]))
        usage = result.usage
        return text, {
            "served_model": result.model, "agent_server": self.target.name,
            "usage": {"input_tokens": getattr(usage, "input_tokens", 0) or 0,
                      "output_tokens": getattr(usage, "output_tokens", 0) or 0},
            "provider_reasoning": reasoning,
        }


class ParleySimEnvironmentServer(
        BaseEnvironmentServer[ParleySimEpisodeRequest, ParleySimEpisodeResponse]):
    """Run one parley-sim episode per request; model seats are Gym agents."""

    config: ParleySimEnvironmentServerConfig
    request_model = ParleySimEpisodeRequest
    response_model = ParleySimEpisodeResponse

    async def aggregate_metrics(self, body: AggregateMetricsRequest = Body()) -> AggregateMetrics:  # noqa: B008
        rewards = [r["reward"] for r in body.verify_responses if r.get("reward") is not None]
        key: dict[str, Any] = {"mean_reward": statistics.fmean(rewards) if rewards else None,
                               "n_episodes": len(body.verify_responses)}
        names = {k for r in body.verify_responses for k in (r.get("metrics") or {})}
        for name in sorted(names):
            vals = [r["metrics"][name] for r in body.verify_responses
                    if (r.get("metrics") or {}).get(name) is not None]
            key[f"mean_{name}"] = statistics.fmean(vals) if vals else None
        return AggregateMetrics(key_metrics=key, agent_metrics=key)

    def responses_path(self, target_name: str, request: ParleySimEpisodeRequest,
                       trained: bool) -> str:
        block = self.server_client.global_config_dict.get(TOKEN_ID_CAPTURE_BLOCK) or {}
        agent = get_first_server_config_dict(self.server_client.global_config_dict, target_name)
        capture = trained and bool(block.get("enabled", False)) and (
            bool(block.get("all_agents", False)) or bool(agent.get("token_id_capture", False)))
        segment = f"/{TOKEN_CAPTURE_PATH_SEGMENT}" if capture else ""
        return f"/ng-rollout/{request.episode_id.capture_key}{segment}/v1/responses"

    async def run(self, request: ParleySimEpisodeRequest,
                  cleanup: CleanupContext) -> ParleySimEpisodeResponse:
        task = request.task.task_input
        try:
            if (":" in task.env and task.env not in envs.registry()
                    and task.env not in self.config.allowed_env_paths):
                raise ValueError(f"env {task.env!r} is an import path not in allowed_env_paths")
            entry = envs.get(task.env)
            if self.config.reward_metric not in entry.metric_kinds:
                raise ValueError(f"reward_metric {self.config.reward_metric!r} is not one of "
                                 f"{sorted(entry.metric_kinds)}")
            focal = task.focal_seat or entry.focal_seat
            spec = EpisodeSpec(seed=task.seed, seats=task.seats, config=task.env_config,
                               focal_seat=focal, arm=task.arm, pair_key=task.pair_key,
                               cell_key=task.cell_key, replicate=task.replicate)
            episode = Episode(entry.env(), spec, scripted_policies(entry, spec))
            seats = self._model_seats(request, cleanup, entry, spec, focal)
        except (ValueError, KeyError) as error:
            raise _failure("seed", f"invalid parley-sim task: {error}", terminal=True) from error

        try:
            event = episode.advance()
            while not isinstance(event, EpisodeComplete):
                event = episode.advance(await seats[event.seat].decide(event))
        except EpisodeAbort as error:
            episode.abort(str(error))
            raise _failure("agent", str(error), terminal=not error.retryable) from error
        except ProviderHalted as error:
            episode.abort(repr(error))
            raise _failure("agent", f"run halted: {error}", terminal=True) from error

        metrics = entry.episode_metrics(episode.lines)
        reward = metrics.get(self.config.reward_metric)
        focal_client = seats[focal].client if focal in seats else None
        return ParleySimEpisodeResponse(
            episode_id=request.episode_id, task_id=request.task.task_id,
            result=ParleySimEpisodeResult(
                reward=float(reward) if reward is not None else 0.0, status=event.status,
                outcomes=event.outcomes, metrics=metrics, n_decisions=event.n_decisions,
                n_strikes=event.n_strikes, trace=episode.lines,
                response=focal_client.training_response() if focal_client else None,
                focal_calls=focal_client.calls if focal_client else []))

    def _model_seats(self, request: ParleySimEpisodeRequest, cleanup: CleanupContext,
                     entry: envs.EnvEntry, spec: EpisodeSpec, focal: Seat) -> dict[Seat, ModelSeat]:
        breaker = Breaker(self.config.max_refusals)
        seats: dict[Seat, ModelSeat] = {}
        for seat, label in spec.seats.items():
            if label.startswith("policy:"):
                continue
            if not label.startswith("model:"):
                raise ValueError(f"seat {seat!r}: label {label!r} must start with "
                                 "'policy:' or 'model:'")
            name = label.split(":", 1)[1]
            if seat == focal:
                target = self.config.focal_agent
            elif name in self.config.counterpart_agents:
                target = self.config.counterpart_agents[name]
            else:
                raise KeyError(f"no counterpart agent configured for {label!r}")
            model = ModelSpec(name=name.replace("/", "-") or "agent", provider="harness",
                              model=target.name,
                              timeout_s=self.config.actor_call_timeout_seconds + 30,
                              conversation=seat == focal and self.config.focal_conversation)
            client = _AgentSessionClient(self, request, cleanup, target, trained=seat == focal)
            seats[seat] = ModelSeat(model, entry.text(), client, breaker)
        return seats


def _failure(stage: str, reason: str, terminal: bool) -> HandledEpisodeError:
    return HandledEpisodeError(EpisodeFailure(stage=stage, failure_reason=reason[:2000],
                                              terminal=terminal))


def _read_output(response: NeMoGymResponse) -> tuple[str, str | None, str | None]:
    """(text, refusal, reasoning summary) from a Responses API result."""
    texts, refusal, reasoning = [], None, []
    for item in response.output:
        data = item.model_dump() if hasattr(item, "model_dump") else dict(item)
        if data.get("type") == "message":
            for part in data.get("content") or []:
                if part.get("type") == "output_text":
                    texts.append(part.get("text", ""))
                elif part.get("type") == "refusal":
                    refusal = part.get("refusal", "")
        elif data.get("type") == "reasoning":
            reasoning += [s.get("text", "") for s in data.get("summary") or []]
    return "".join(texts), refusal, "\n".join(reasoning) or None


def _describe(error: BaseException) -> str:
    try:
        return f"{type(error).__name__}: {error}"
    except Exception:
        return type(error).__name__


def _cookies(response: Any) -> dict[str, str]:
    return {str(name): str(morsel.value) for name, morsel in response.cookies.items()}


def _is_retryable(error: BaseException) -> bool:
    if isinstance(error, ClientResponseError):
        return error.status in {408, 425, 429} or error.status >= 500
    return isinstance(error, (ClientConnectionError, ClientPayloadError, TimeoutError))


__all__ = ["ParleySimEnvironmentServer", "ParleySimEnvironmentServerConfig",
           "ParleySimEpisodeRequest", "ParleySimEpisodeResponse", "ParleySimEpisodeResult",
           "ParleySimTaskInput"]
