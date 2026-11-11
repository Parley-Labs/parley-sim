"""OpenEnv environment for parley-sim episodes (`pip install parley-sim[openenv]`).

The OpenEnv client (a trainer such as TRL's GRPOTrainer, or a script) plays the
focal seat. `reset` builds one parley-sim Episode and advances every other seat
until the focal seat must act; `step` submits the focal seat's move and does
the same again. Scripted counterparts run in-process; model counterparts are
parley-sim model seats called from inside the environment. The full trace,
including every counterpart decision, is exposed through `state`, so opponent
behavior is never hidden in the container.

Targets openenv 0.12. A server entry point (OpenEnv's `server/app.py`):

    from parley_sim.adapters.openenv import create_parley_app

    app = create_parley_app(
        seats={"agent": "client", "customer": "policy:hardliner"},
    )

    # uvicorn app:app --port 8000

and a client loop, here driving the focal seat with any model:

    from parley_sim.adapters.openenv import ParleySimAction, ParleySimEnv

    with ParleySimEnv(base_url="http://localhost:8000").sync() as env:
        result = env.reset(seed=1000)
        while not result.done:
            obs = result.observation
            reply = my_model(obs.system_prompt, obs.user_prompt)  # JSON per obs.action_schema
            result = env.step(ParleySimAction(text=reply))
        print(result.reward, env.state().metrics)
"""
from __future__ import annotations

import asyncio
from typing import Any, Callable
from uuid import uuid4

from openenv.core.client_types import StepResult
from openenv.core.env_client import EnvClient
from openenv.core.env_server.http_server import create_app
from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import Action, Observation, State
from pydantic import Field, model_validator

from parley_sim import envs
from parley_sim.core.protocol import JSON, MalformedOutput, Seat
from parley_sim.drivers.models import (Breaker, ModelSeat, ModelSpec, ProviderClient,
                                       ScriptedSeat, build_client)
from parley_sim.runtime.episode import (DecisionRequest, DecisionResult, Episode,
                                        EpisodeComplete, EpisodeSpec)

CLIENT = "client"


class ParleySimAction(Action):
    """The focal seat's turn: either the model's raw reply (`text`, parsed with
    the env's text interface) or an already-structured `move`."""

    text: str | None = Field(default=None, description="Model output, JSON per action_schema")
    move: dict[str, Any] | None = Field(default=None, description="Structured env action")

    @model_validator(mode="after")
    def _one(self) -> "ParleySimAction":
        if (self.text is None) == (self.move is None):
            raise ValueError("give exactly one of text or move")
        return self


class ParleySimObservation(Observation):
    seat: str | None = Field(default=None, description="Seat that must act next")
    turn: int = 0
    system_prompt: str = ""
    user_prompt: str = ""
    action_schema: dict[str, Any] = Field(default_factory=dict)
    observation: dict[str, Any] = Field(default_factory=dict, description="Raw env view")
    last_strike: str | None = Field(default=None,
                                    description="Why the previous focal action fell back")
    outcomes: dict[str, float] = Field(default_factory=dict)
    metrics: dict[str, Any] = Field(default_factory=dict)


class ParleySimState(State):
    seed: int | None = None
    arm: str | None = None
    focal_seat: str | None = None
    seats: dict[str, str] = Field(default_factory=dict)
    done: bool = False
    outcomes: dict[str, float] = Field(default_factory=dict)
    metrics: dict[str, Any] = Field(default_factory=dict)
    trace: list[dict[str, Any]] = Field(default_factory=list)


class ParleySimEnvironment(Environment[ParleySimAction, ParleySimObservation, ParleySimState]):
    """One instance per OpenEnv session. `seats` labels each seat: the focal
    seat is "client" (played through step), others "policy:<name>" or
    "model:<lineup name>"."""

    SUPPORTS_CONCURRENT_SESSIONS = True

    def __init__(self, *, env: str = "support_desk", seats: dict[Seat, str],
                 focal_seat: Seat | None = None, env_config: dict[str, Any] | None = None,
                 lineup: list[ModelSpec | dict[str, Any]] | None = None,
                 client_factory: Callable[[ModelSpec], ProviderClient] = build_client,
                 reward_metric: str = "focal_surplus_share", max_refusals: int = 3):
        super().__init__()
        self.entry = envs.get(env)
        self.focal = focal_seat or self.entry.focal_seat
        if seats.get(self.focal) != CLIENT:
            raise ValueError(f"seat {self.focal!r} must be labelled {CLIENT!r}")
        self.seats = dict(seats)
        self.env_config = dict(env_config or {})
        self.lineup = {m.name: m for m in (ModelSpec(**m) if isinstance(m, dict) else m
                                           for m in lineup or [])}
        self.client_factory = client_factory
        self.reward_metric = reward_metric
        self.max_refusals = max_refusals
        self.text = self.entry.text()
        self._episode: Episode | None = None
        self._pending: DecisionRequest | None = None
        self._counterparts: dict[Seat, ModelSeat | ScriptedSeat] = {}
        self._state = ParleySimState()

    def reset(self, seed: int | None = None, episode_id: str | None = None,
              arm: str | None = None, **kwargs: Any) -> ParleySimObservation:
        return _run(self.reset_async(seed=seed, episode_id=episode_id, arm=arm))

    async def reset_async(self, seed: int | None = None, episode_id: str | None = None,
                          arm: str | None = None, **kwargs: Any) -> ParleySimObservation:
        seed = 0 if seed is None else int(seed)
        spec = EpisodeSpec(seed=seed, seats=self.seats, config=self.env_config,
                           focal_seat=self.focal, arm=arm, cell_key=episode_id)
        policies = {s: self.entry.make_policy(label.split(":", 1)[1], s)
                    for s, label in self.seats.items() if label.startswith("policy:")}
        self._episode = Episode(self.entry.env(), spec, policies)
        breaker = Breaker(self.max_refusals)
        self._counterparts = {}
        for s, label in self.seats.items():
            if s == self.focal or not label.startswith("model:"):
                continue
            model = self.lineup[label.split(":", 1)[1]]
            if model.provider == "scripted":
                self._counterparts[s] = ScriptedSeat(model, self.text,
                                                     self.entry.make_policy(model.model, s))
            else:
                self._counterparts[s] = ModelSeat(model, self.text, self.client_factory(model),
                                                  breaker)
        self._state = ParleySimState(episode_id=episode_id or str(uuid4()), step_count=0,
                                     seed=seed, arm=arm, focal_seat=self.focal,
                                     seats=self.seats)
        return await self._advance(self._episode.advance(), strike=None)

    def step(self, action: ParleySimAction, timeout_s: float | None = None,
             **kwargs: Any) -> ParleySimObservation:
        return _run(self.step_async(action, timeout_s=timeout_s))

    async def step_async(self, action: ParleySimAction, timeout_s: float | None = None,
                         **kwargs: Any) -> ParleySimObservation:
        if self._episode is None or self._pending is None:
            raise RuntimeError("call reset() before step(), and not after the episode ends")
        req, self._pending = self._pending, None
        self._state.step_count += 1
        if action.text is not None:
            try:
                parsed, explanation = self.text.parse(action.text)
                result = DecisionResult(req.request_id, parsed, explanation=explanation,
                                        raw={"source": "openenv", "payload": action.text})
            except MalformedOutput as exc:
                result = DecisionResult(req.request_id, None, strike="malformed_output",
                                        detail=str(exc),
                                        raw={"source": "openenv", "payload": action.text})
        else:
            result = DecisionResult(req.request_id, action.move, raw={"source": "openenv"})
        event = self._episode.advance(result)
        decision = next(l for l in reversed(self._episode.lines) if l["type"] == "decision"
                        and l["seat"] == self.focal)
        return await self._advance(event, strike=decision["strike"])

    @property
    def state(self) -> ParleySimState:
        if self._episode is not None:
            self._state.trace = list(self._episode.lines)
        return self._state

    async def _advance(self, event: DecisionRequest | EpisodeComplete,
                       strike: str | None) -> ParleySimObservation:
        assert self._episode is not None
        try:
            while isinstance(event, DecisionRequest) and event.seat != self.focal:
                event = self._episode.advance(await self._counterparts[event.seat].decide(event))
        except BaseException as exc:
            if self._episode.result is None:
                self._episode.abort(repr(exc))
            self._state.done = True
            raise
        if isinstance(event, EpisodeComplete):
            metrics = self.entry.episode_metrics(self._episode.lines)
            reward = metrics.get(self.reward_metric)
            self._state.done, self._state.outcomes, self._state.metrics = (
                True, dict(event.outcomes), metrics)
            return ParleySimObservation(done=True, reward=float(reward or 0.0),
                                        last_strike=strike, outcomes=dict(event.outcomes),
                                        metrics=metrics,
                                        turn=self._episode.env.turn(self._episode.state))
        self._pending = event
        system, user = self.text.render(event.observation)
        return ParleySimObservation(seat=event.seat, turn=event.turn, system_prompt=system,
                                    user_prompt=user, action_schema=self.text.action_schema,
                                    observation=event.observation, last_strike=strike)


def _run(coro: Any) -> Any:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    coro.close()
    raise RuntimeError("an event loop is running here: use reset_async/step_async")


def create_parley_app(*, max_concurrent_envs: int = 8, **environment_kwargs: Any):
    """An OpenEnv FastAPI app serving ParleySimEnvironment(**environment_kwargs)."""
    return create_app(lambda: ParleySimEnvironment(**environment_kwargs), ParleySimAction,
                      ParleySimObservation, env_name="parley_sim",
                      max_concurrent_envs=max_concurrent_envs, state_cls=ParleySimState)


class ParleySimEnv(EnvClient[ParleySimAction, ParleySimObservation, ParleySimState]):
    """OpenEnv client for a parley-sim server."""

    def _step_payload(self, action: ParleySimAction) -> JSON:
        return action.model_dump(exclude_none=True, exclude={"metadata"})

    def _parse_result(self, payload: JSON) -> StepResult[ParleySimObservation]:
        obs = dict(payload.get("observation", {}))
        obs.pop("metadata", None)
        observation = ParleySimObservation(**obs, reward=payload.get("reward"),
                                           done=payload.get("done", False),
                                           metadata=payload.get("metadata") or {})
        return StepResult(observation=observation, reward=payload.get("reward"),
                          done=payload.get("done", False), metadata=payload.get("metadata"))

    def _parse_state(self, payload: JSON) -> ParleySimState:
        return ParleySimState(**payload)
