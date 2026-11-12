"""OpenEnv environment for parley-sim episodes (`pip install parley-sim[openenv]`).

The OpenEnv client (a trainer such as TRL's GRPOTrainer, or a script) plays the
focal seat. `reset` builds one parley-sim Episode and advances every other seat
until the focal seat must act; `step` submits the focal seat's move and does
the same again. Scripted counterparts run in-process; model counterparts are
parley-sim model seats called from inside the environment.

While an episode runs, `state` shows only what the focal seat can see. Once the
episode is done, its full trace (the scenario, ground truth and every
counterpart decision) is released through `state` by default, so opponent
behavior is never hidden in the container. A client chooses its seeds, so it can
end an episode at once, read the released trace, and replay the same seed: when
the client must not learn counterpart information, serve with
`release_trace="never"` (`PARLEY_RELEASE_TRACE=never` in the packaged server).

Play goes only through OpenEnv's WebSocket sessions (`/ws`, which `ParleySimEnv`
uses), where the session limit, idle timeout and refusal breaker apply. The
stateless HTTP `/reset`, `/step` and `/state` routes are disabled, and the
Gradio web interface is never mounted.

Targets openenv 0.12. A server entry point (OpenEnv's `server/app.py`):

    from parley_sim.adapters.openenv import create_parley_app

    app = create_parley_app(
        seats={"agent": "client", "customer": "policy:hardliner"},
    )

    # uvicorn app:app --host 127.0.0.1 --port 8000

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
import functools
import inspect
import json
import logging
import os
from collections.abc import Callable
from typing import Any
from uuid import uuid4

from fastapi import Request
from fastapi.responses import JSONResponse
from fastapi.routing import APIRoute
from openenv.core.client_types import StepResult
from openenv.core.env_client import EnvClient
from openenv.core.env_server.http_server import create_fastapi_app
from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import Action, ConcurrencyConfig, Observation, State
from pydantic import Field, model_validator

from parley_sim import envs
from parley_sim.core.protocol import JSON, MalformedOutput, Seat
from parley_sim.drivers.models import (
    Breaker,
    ModelSeat,
    ModelSpec,
    ProviderClient,
    ScriptedSeat,
    build_client,
)
from parley_sim.runtime.episode import DecisionRequest, DecisionResult, Episode, EpisodeComplete, EpisodeSpec

CLIENT = "client"
RELEASE_TRACE = ("after_done", "never")
# Longest focal action accepted, in characters (`text`, or `move` as JSON). Every
# action is stored in the trace and sent back with each state().
MAX_ACTION_CHARS = 65536
# Client wait for one server reply. One counterpart decision can take two
# attempts of the model's timeout_s (120 s by default), so OpenEnv's 60 s is too
# short.
DEFAULT_MESSAGE_TIMEOUT_S = 300.0
# Anything a text interface's parse can raise on hostile input: deeply nested
# JSON overflows the recursion limit, odd types raise TypeError.
_PARSE_ERRORS = (MalformedOutput, RecursionError, ValueError, TypeError)
_STATELESS_ROUTES = ("/reset", "/step", "/state")
_log = logging.getLogger(__name__)
_closing: set[asyncio.Task[None]] = set()  # strong refs to scheduled client closes


class ParleySimAction(Action):
    """The focal seat's turn: either the model's raw reply (`text`, parsed with
    the env's text interface) or an already-structured `move`. A `move` is cut
    down to the keys the action schema declares and then parsed like `text`."""

    text: str | None = Field(default=None, max_length=MAX_ACTION_CHARS,
                             description="Model output, JSON per action_schema")
    move: dict[str, Any] | None = Field(default=None, description="Structured env action")

    @model_validator(mode="after")
    def _one(self) -> ParleySimAction:
        if (self.text is None) == (self.move is None):
            raise ValueError("give exactly one of text or move")
        if self.move is not None:
            try:
                size = len(json.dumps(self.move))
            except (RecursionError, TypeError, ValueError):
                raise ValueError("move must be plain, not deeply nested, JSON") from None
            if size > MAX_ACTION_CHARS:
                raise ValueError(f"move is {size} characters as JSON; "
                                 f"the limit is {MAX_ACTION_CHARS}")
        return self


def _project(value: Any, schema: Any) -> Any:
    """`value` with only the object keys `schema` declares, recursively."""
    if not isinstance(schema, dict):
        return value
    props = schema.get("properties")
    if isinstance(value, dict) and isinstance(props, dict):
        return {k: _project(v, props[k]) for k, v in value.items() if k in props}
    items = schema.get("items")
    if isinstance(value, list) and isinstance(items, dict):
        return [_project(v, items) for v in value]
    return value


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
    "model:<lineup name>".

    `state().trace` holds only what the focal seat may see (its own decisions
    and the public move events) until the episode ends. With
    `release_trace="after_done"` (the default) the full trace, with the
    scenario, ground truth and counterpart decisions, is released once the
    episode is done. A client picks its seeds, so it can end an episode at once,
    read that trace, and replay the seed: use `release_trace="never"` when the
    client must not learn counterpart information. Then the trace stays
    focal-only for good, and `state()` adds only the outcomes and metrics.
    The refusal breaker spans the whole session."""

    SUPPORTS_CONCURRENT_SESSIONS = True

    def __init__(self, *, env: str = "support_desk", seats: dict[Seat, str],
                 focal_seat: Seat | None = None, env_config: dict[str, Any] | None = None,
                 lineup: list[ModelSpec | dict[str, Any]] | None = None,
                 client_factory: Callable[[ModelSpec], ProviderClient] = build_client,
                 reward_metric: str = "focal_surplus_share", max_refusals: int = 3,
                 release_trace: str = "after_done"):
        super().__init__()
        self._clients: dict[str, ProviderClient] = {}
        self._loop: asyncio.AbstractEventLoop | None = None  # the sync API's loop
        self._client_loop: asyncio.AbstractEventLoop | None = None  # where clients ran
        self._closed = False
        self._episode: Episode | None = None
        self._pending: DecisionRequest | None = None
        self.entry = envs.get(env)
        self.focal = focal_seat or self.entry.focal_seat
        if seats.get(self.focal) != CLIENT:
            raise ValueError(f"seat {self.focal!r} must be labelled {CLIENT!r}")
        if release_trace not in RELEASE_TRACE:
            raise ValueError(f"release_trace {release_trace!r} must be one of {list(RELEASE_TRACE)}")
        self.release_trace = release_trace
        self.seats = dict(seats)
        self.env_config = dict(env_config or {})
        self.lineup = {m.name: m for m in (ModelSpec(**m) if isinstance(m, dict) else m
                                           for m in lineup or [])}
        self.client_factory = client_factory
        if reward_metric not in self.entry.metric_kinds:
            raise ValueError(f"reward_metric {reward_metric!r} is not one of "
                             f"{sorted(self.entry.metric_kinds)}")
        self.reward_metric = reward_metric
        self.max_refusals = max_refusals
        self._breaker = Breaker(max_refusals)
        try:
            for seat, label in self.seats.items():  # fail at startup, not at the first reset
                if seat == self.focal:
                    continue
                kind, _, name = label.partition(":")
                if kind == "policy":
                    self.entry.make_policy(name, seat)
                elif kind == "model":
                    if name not in self.lineup:
                        raise ValueError(f"seat {seat!r}: {name!r} is not in the lineup")
                    model = self.lineup[name]
                    if model.provider == "scripted":
                        self.entry.make_policy(model.model, seat)
                    elif name not in self._clients:
                        self._clients[name] = client_factory(model)
                else:
                    raise ValueError(f"seat {seat!r}: label {label!r} must start with "
                                     "'policy:' or 'model:'")
        except BaseException:
            self.close()  # clients made for earlier seats
            raise
        self.text = self.entry.text()
        self._counterparts: dict[Seat, ModelSeat | ScriptedSeat] = {}
        self._state = ParleySimState()

    @property
    def max_decision_s(self) -> float:
        """Longest one counterpart decision can take: two attempts of the
        slowest seated model's timeout_s (0 with no model counterparts)."""
        models = [self.lineup[label.split(":", 1)[1]] for seat, label in self.seats.items()
                  if seat != self.focal and label.startswith("model:")]
        return 2 * max((m.timeout_s for m in models if m.provider != "scripted"), default=0.0)

    def close(self) -> None:
        """Called by OpenEnv when a session ends or is reaped for idleness.
        After this, reset and step are refused, so a connection whose slot
        was released can make no further episodes or provider calls. Closes
        the provider clients; calling it again does nothing."""
        self._closed = True
        self._episode, self._pending = None, None
        clients, self._clients = list(self._clients.values()), {}
        if clients:
            self._close_clients(clients)
        if self._loop is not None and not self._loop.is_closed() and not self._loop.is_running():
            self._loop.close()

    def _close_clients(self, clients: list[ProviderClient]) -> None:
        """Close the clients on the loop they ran on, which owns their
        connections. OpenEnv calls close() from a worker thread while the
        server's loop runs, or from that loop itself; the sync API has its own
        persistent loop; a client that never ran can close on any loop."""
        coro = _aclose_all(clients)
        loop = self._client_loop
        if loop is not None and loop.is_closed():
            loop = None
        try:
            running: asyncio.AbstractEventLoop | None = asyncio.get_running_loop()
        except RuntimeError:
            running = None
        if loop is not None and loop.is_running() and loop is not running:
            future = asyncio.run_coroutine_threadsafe(coro, loop)
            if running is None:  # a worker thread: wait, the loop is free to run it
                try:
                    future.result(timeout=10)
                except Exception as exc:  # best effort, like OpenEnv's own cleanup
                    _log.warning("closing provider clients failed: %r", exc)
        elif running is not None:
            task = running.create_task(coro)
            _closing.add(task)
            task.add_done_callback(_closing.discard)
        elif loop is not None:
            loop.run_until_complete(coro)
        else:
            asyncio.run(coro)

    def _check_open(self) -> None:
        if self._closed:
            raise RuntimeError("this session was closed (ended or idle too long); open a new one")

    def _run(self, coro: Any) -> Any:
        """Run a coroutine for the synchronous API on one persistent loop, so
        provider clients are never reused across closed event loops."""
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            if self._loop is None or self._loop.is_closed():
                self._loop = asyncio.new_event_loop()
            return self._loop.run_until_complete(coro)
        coro.close()
        raise RuntimeError("an event loop is running here: use reset_async/step_async")

    def reset(self, seed: int | None = None, episode_id: str | None = None,
              arm: str | None = None, **kwargs: Any) -> ParleySimObservation:
        return self._run(self.reset_async(seed=seed, episode_id=episode_id, arm=arm))

    async def reset_async(self, seed: int | None = None, episode_id: str | None = None,
                          arm: str | None = None, **kwargs: Any) -> ParleySimObservation:
        self._check_open()
        self._client_loop = asyncio.get_running_loop()
        # Build and validate everything first, so a bad argument leaves the
        # previous episode and its state as they were.
        seed = 0 if seed is None else int(seed)
        state = ParleySimState(episode_id=episode_id or str(uuid4()), step_count=0,
                               seed=seed, arm=arm, focal_seat=self.focal, seats=self.seats)
        spec = EpisodeSpec(seed=seed, seats=self.seats, config=self.env_config,
                           focal_seat=self.focal, arm=arm, cell_key=episode_id)
        policies = {s: self.entry.make_policy(label.split(":", 1)[1], s)
                    for s, label in self.seats.items() if label.startswith("policy:")}
        episode = Episode(self.entry.env(), spec, policies)
        counterparts: dict[Seat, ModelSeat | ScriptedSeat] = {}
        for s, label in self.seats.items():
            if s == self.focal or not label.startswith("model:"):
                continue
            model = self.lineup[label.split(":", 1)[1]]
            if model.provider == "scripted":
                counterparts[s] = ScriptedSeat(model, self.text,
                                               self.entry.make_policy(model.model, s))
            else:
                counterparts[s] = ModelSeat(model, self.text, self._clients[model.name],
                                            self._breaker)
        first = episode.advance()
        self._episode, self._counterparts, self._state, self._pending = (
            episode, counterparts, state, None)
        return await self._advance(first, strike=None)

    def step(self, action: ParleySimAction, timeout_s: float | None = None,
             **kwargs: Any) -> ParleySimObservation:
        return self._run(self.step_async(action, timeout_s=timeout_s))

    async def step_async(self, action: ParleySimAction, timeout_s: float | None = None,
                         **kwargs: Any) -> ParleySimObservation:
        self._check_open()
        self._client_loop = asyncio.get_running_loop()
        episode, req = self._episode, self._pending
        if episode is None or req is None:
            raise RuntimeError("call reset() before step(), and not after the episode ends")
        # The request stays pending until the episode takes the result, so an
        # unexpected error before that leaves the session able to step again.
        result = self._focal_result(req, action)
        try:
            event = episode.advance(result)
        except BaseException as exc:  # the episode took the request: it cannot go on
            self._pending = None
            if episode.result is None:
                episode.abort(repr(exc))
            self._state.done = True
            raise
        self._pending = None
        self._state.step_count += 1
        decision = next(l for l in reversed(episode.lines) if l["type"] == "decision"
                        and l["seat"] == self.focal)
        return await self._advance(event, strike=decision["strike"])

    def _focal_result(self, req: DecisionRequest, action: ParleySimAction) -> DecisionResult:
        """Parse the focal action as a model seat's reply would be. A `move`
        keeps only the keys the action schema declares. Any parse failure is
        a malformed_output strike."""
        if action.text is not None:
            payload: Any = action.text
            raw: JSON = {"source": "openenv", "payload": action.text}
        else:
            payload, raw = _project(action.move, self.text.action_schema), {"source": "openenv"}
        try:
            parsed, explanation = self.text.parse(payload)
        except _PARSE_ERRORS as exc:
            detail = str(exc) if isinstance(exc, MalformedOutput) else f"{type(exc).__name__}: {exc}"
            return DecisionResult(req.request_id, None, strike="malformed_output",
                                  detail=detail[:500], raw=raw)
        return DecisionResult(req.request_id, parsed, explanation=explanation, raw=raw)

    @property
    def state(self) -> ParleySimState:
        if self._episode is not None:
            lines = self._episode.lines
            if self._episode.result is None or self.release_trace == "never":
                # Only what the focal seat can see. A decision's `context` is
                # evaluation data (it can hold the counterpart's valuation of a
                # standing offer), so it is left out.
                lines = [l if l["type"] == "event" else {k: v for k, v in l.items() if k != "context"}
                         for l in lines if l["type"] == "event"
                         or (l["type"] == "decision" and l["seat"] == self.focal)]
            self._state.trace = list(lines)
        return self._state

    async def _advance(self, event: DecisionRequest | EpisodeComplete,
                       strike: str | None) -> ParleySimObservation:
        episode = self._episode
        assert episode is not None
        try:
            while isinstance(event, DecisionRequest) and event.seat != self.focal:
                decision = await self._counterparts[event.seat].decide(event)
                self._check_open()  # the idle reaper may have closed the session meanwhile
                event = episode.advance(decision)
        except BaseException as exc:
            if episode.result is None:
                episode.abort(repr(exc))
            self._state.done = True
            raise
        if isinstance(event, EpisodeComplete):
            metrics = self.entry.episode_metrics(episode.lines)
            reward = metrics.get(self.reward_metric)
            self._state.done, self._state.outcomes, self._state.metrics = (
                True, dict(event.outcomes), metrics)
            return ParleySimObservation(done=True, reward=float(reward or 0.0),
                                        last_strike=strike, outcomes=dict(event.outcomes),
                                        metrics=metrics,
                                        turn=episode.env.turn(episode.state))
        self._pending = event
        system, user = self.text.render(event.observation)
        return ParleySimObservation(seat=event.seat, turn=event.turn, system_prompt=system,
                                    user_prompt=user, action_schema=self.text.action_schema,
                                    observation=event.observation, last_strike=strike)


async def _aclose_all(clients: list[ProviderClient]) -> None:
    """Close each provider client: its own aclose/close if it has one, else
    the SDK client it wraps (`_client`, an AsyncOpenAI or AsyncAnthropic)."""
    for client in clients:
        closer = (getattr(client, "aclose", None) or getattr(client, "close", None)
                  or getattr(getattr(client, "_client", None), "close", None))
        if closer is None:
            continue
        try:
            result = closer()
            if inspect.isawaitable(result):
                await result
        except Exception as exc:  # best effort: one failure must not keep the others open
            _log.warning("closing provider client %r failed: %r", client, exc)


def _sessions_only(request: Request) -> JSONResponse:
    return JSONResponse(status_code=405, content={"detail": (
        f"{request.url.path} is disabled on this server: stateless HTTP calls bypass the "
        "session limit, idle timeout and refusal breaker. Play over the WebSocket session "
        "API at /ws (ParleySimEnv does).")})


def create_parley_app(*, max_concurrent_envs: int = 8, session_timeout_s: float | None = 600.0,
                      **environment_kwargs: Any):
    """An OpenEnv FastAPI app serving ParleySimEnvironment(**environment_kwargs).

    All play goes through WebSocket sessions (`/ws`): the stateless HTTP
    `/reset`, `/step` and `/state` routes answer 405, since each such call
    would build a fresh environment (new breaker, new provider clients)
    outside the session limit. The Gradio web interface is never mounted,
    whatever ENABLE_WEB_INTERFACE says, because it shares one environment
    between all its users. Idle sessions are closed after `session_timeout_s`,
    which must cover two attempts of the slowest model counterpart's
    timeout_s. The server has no authentication of its own."""
    probe = ParleySimEnvironment(**environment_kwargs)  # validate the configuration at startup
    try:
        needed = probe.max_decision_s
    finally:
        probe.close()
    if session_timeout_s is not None and session_timeout_s < needed:
        raise ValueError(
            f"session timeout {session_timeout_s:g}s (PARLEY_SESSION_TIMEOUT) is shorter than "
            f"one counterpart decision can take ({needed:g}s: two attempts of the slowest "
            "model's timeout_s), so a session could be closed as idle mid-step. Raise the "
            f"session timeout to at least {needed:g}s, or lower timeout_s in the lineup.")
    if os.getenv("ENABLE_WEB_INTERFACE", "false").strip().lower() in ("true", "1", "yes"):
        _log.warning("ENABLE_WEB_INTERFACE is set, but parley-sim never mounts OpenEnv's web "
                     "interface: it would share one environment between all its users. Serve "
                     "without it (`openenv push --no-interface`) and play over /ws.")
    # A partial over the class, not a lambda: OpenEnv then reads the class
    # attributes instead of building (and closing) a throwaway environment.
    factory = functools.partial(ParleySimEnvironment, **environment_kwargs)
    app = create_fastapi_app(factory, ParleySimAction, ParleySimObservation, env_name="parley_sim",
                             concurrency_config=ConcurrencyConfig(
                                 max_concurrent_envs=max_concurrent_envs,
                                 session_timeout=session_timeout_s),
                             state_cls=ParleySimState)
    app.router.routes[:] = [r for r in app.router.routes
                            if not (isinstance(r, APIRoute) and r.path in _STATELESS_ROUTES)]
    for path in _STATELESS_ROUTES:
        app.add_api_route(path, _sessions_only, methods=["GET", "POST"], include_in_schema=False)
    return app


class ParleySimEnv(EnvClient[ParleySimAction, ParleySimObservation, ParleySimState]):
    """OpenEnv client for a parley-sim server. It waits up to
    `message_timeout_s` (default 300 s; OpenEnv's own default is 60 s) for each
    reply, since one counterpart decision can take two attempts of the model's
    timeout_s (120 s each by default). Raise it if you raise timeout_s."""

    def __init__(self, base_url: str | None = None, connect_timeout_s: float = 10.0,
                 message_timeout_s: float = DEFAULT_MESSAGE_TIMEOUT_S, **kwargs: Any):
        super().__init__(base_url, connect_timeout_s, message_timeout_s, **kwargs)

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
