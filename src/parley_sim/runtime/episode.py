"""Inverted-control episode runtime.

The runtime never calls a model. Scripted seats are resolved in-process;
whenever a non-scripted seat must act, `advance` returns a DecisionRequest and
waits for the caller to answer it. Every driver (the local runner, a NeMo Gym
environment server, an OpenEnv environment) is a loop around:

    event = episode.advance()
    while not isinstance(event, EpisodeComplete):
        event = episode.advance(answer(event))
"""
from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, field
from typing import Any

from parley_sim.core.hashing import digest
from parley_sim.core.protocol import JSON, Environment, IllegalAction, Policy, Seat
from parley_sim.trace import SCHEMA_VERSION

Sink = Callable[[dict[str, Any]], None]


@dataclass(frozen=True)
class EpisodeSpec:
    seed: int
    seats: dict[Seat, str]  # seat -> "policy:<name>" | "model:<name>" | any external label
    config: dict[str, Any] = field(default_factory=dict)
    focal_seat: Seat | None = None
    arm: str | None = None
    pair_key: str | None = None
    cell_key: str | None = None
    replicate: int = 0


@dataclass(frozen=True)
class DecisionRequest:
    request_id: str
    seat: Seat
    turn: int
    observation: JSON
    context: JSON


@dataclass(frozen=True)
class DecisionResult:
    request_id: str
    action: JSON | None = None
    reasoning: str | None = None  # provider-returned deliberation (e.g. thinking summary)
    strike: str | None = None  # why no usable action: "timeout", "malformed_output", ...
    detail: str | None = None
    raw: JSON | None = None  # adapter-defined: request, response, usage, latency
    explanation: str | None = None  # the model's own stated reason, in its output


@dataclass(frozen=True)
class EpisodeComplete:
    status: str  # "complete" | "errored"
    outcomes: dict[Seat, float]
    n_decisions: int
    n_strikes: int


class Episode:
    def __init__(self, env: Environment, spec: EpisodeSpec,
                 policies: Mapping[Seat, Policy], sink: Sink | None = None):
        self.env = env
        self.spec = spec
        self.lines: list[dict[str, Any]] = []
        self._sink = sink
        self.instance = env.generate(spec.seed, spec.config)
        seats = set(env.seats(self.instance))
        if set(spec.seats) != seats:
            raise ValueError(f"spec seats {sorted(spec.seats)} != env seats {sorted(seats)}")
        unknown = set(policies) - seats
        if unknown:
            raise ValueError(f"policies for unknown seats: {sorted(unknown)}")
        for seat, label in spec.seats.items():
            if label.startswith("policy:") and seat not in policies:
                raise ValueError(f"seat {seat!r} is {label!r} but no policy was given")
        self.policies = dict(policies)
        self.state = env.initial_state(self.instance)
        self._seq = 0
        self._n_decisions = 0
        self._n_strikes = 0
        self._pending: DecisionRequest | None = None
        self.result: EpisodeComplete | None = None
        self._emit({
            "type": "header",
            "schema_version": SCHEMA_VERSION,
            "env": {"name": env.name, "version": env.version},
            "spec": asdict(spec),
            "policies": {s: {"name": p.name, "params": p.params}
                         for s, p in sorted(self.policies.items())},
            "instance": env.instance_to_json(self.instance),
            "ground_truth": env.ground_truth(self.instance),
            "initial_state_hash": digest(env.state_to_json(self.state)),
        })

    def advance(self, result: DecisionResult | None = None) -> DecisionRequest | EpisodeComplete:
        if self.result is not None:
            raise RuntimeError("episode already finished")
        if self._pending is not None:
            if result is None or result.request_id != self._pending.request_id:
                raise ValueError(f"expected a result for {self._pending.request_id!r}")
            pending, self._pending = self._pending, None
            self._take_external(pending, result)
        elif result is not None:
            raise ValueError("no decision is pending")

        while True:
            seat = self.env.to_act(self.state)
            if seat is None:
                return self._finish("complete")
            observation = self.env.observe(self.state, seat)
            context = self.env.decision_context(self.state, seat)
            policy = self.policies.get(seat)
            if policy is None:
                self._pending = DecisionRequest(
                    request_id=f"{self._seq}:{seat}", seat=seat,
                    turn=self.env.turn(self.state), observation=observation, context=context)
                return self._pending
            action = policy.decide(observation)
            self._commit(seat, observation, context, source="scripted", submitted=action,
                         applied=action, reasoning=None, explanation=None, strike=None,
                         detail=None, raw=None)

    def abort(self, error: str) -> EpisodeComplete:
        """Close the trace with an errored footer (call from a driver's except)."""
        return self._finish("errored", error=error)

    def _take_external(self, req: DecisionRequest, result: DecisionResult) -> None:
        strike, detail, applied = result.strike, result.detail, result.action
        if applied is None and strike is None:
            strike = "no_action"
        if applied is not None:
            try:
                self.env.apply(self.state, req.seat, applied)
            except IllegalAction as exc:
                strike, detail, applied = "illegal_action", str(exc), None
        if applied is None:
            applied = self.env.fallback_action(self.state, req.seat)
        self._commit(req.seat, req.observation, req.context, source="external",
                     submitted=result.action, applied=applied, reasoning=result.reasoning,
                     explanation=result.explanation, strike=strike, detail=detail,
                     raw=result.raw)

    def _commit(self, seat: Seat, observation: JSON, context: JSON, *, source: str,
                submitted: JSON | None, applied: JSON, reasoning: str | None,
                explanation: str | None, strike: str | None, detail: str | None,
                raw: JSON | None) -> None:
        turn = self.env.turn(self.state)
        self.state, events = self.env.apply(self.state, seat, applied)
        self._n_decisions += 1
        self._n_strikes += strike is not None
        self._emit({
            "type": "decision", "turn": turn, "seat": seat, "source": source,
            "label": self.spec.seats[seat], "observation": observation, "context": context,
            "submitted": submitted, "applied": applied, "reasoning": reasoning,
            "explanation": explanation, "strike": strike, "detail": detail, "raw": raw,
        })
        state_hash = digest(self.env.state_to_json(self.state))
        for ev in events:
            self._emit({"type": "event", "turn": turn, "actor": seat, **ev,
                        "state_hash": state_hash})

    def _finish(self, status: str, error: str | None = None) -> EpisodeComplete:
        outcomes = self.env.outcome(self.state) if status == "complete" else {}
        self.result = EpisodeComplete(status=status, outcomes=outcomes,
                                      n_decisions=self._n_decisions, n_strikes=self._n_strikes)
        footer = {"type": "footer", "status": status, "outcomes": outcomes,
                  "n_decisions": self._n_decisions, "n_strikes": self._n_strikes,
                  "final_state": self.env.state_to_json(self.state),
                  "final_state_hash": digest(self.env.state_to_json(self.state))}
        if error is not None:
            footer["error"] = error
        self._emit(footer)
        return self.result

    def _emit(self, line: dict[str, Any]) -> None:
        line = {"seq": self._seq, **line}
        self._seq += 1
        self.lines.append(line)
        if self._sink is not None:
            self._sink(line)


def verify(env: Environment, lines: list[dict[str, Any]]) -> list[str]:
    """Re-fold a recorded trace through the pure environment. Returns a list of
    mismatches (empty means the trace is exactly reproducible): the scenario,
    ground truth, every state hash, the final state and the outcomes."""
    if not lines or lines[0].get("type") != "header":
        return ["no header"]
    header = lines[0]
    if lines[-1].get("type") != "footer":
        return ["no footer: the trace is truncated"]
    footer = lines[-1]
    spec = header["spec"]
    problems: list[str] = []
    if header.get("env", {}).get("version") not in (None, env.version):
        problems.append(f"env version {header['env']['version']} != {env.version}")
    instance = env.generate(spec["seed"], spec["config"])
    if env.instance_to_json(instance) != header["instance"]:
        problems.append("instance differs from header")
    if env.ground_truth(instance) != header["ground_truth"]:
        problems.append("ground truth differs from header")
    state = env.initial_state(instance)
    last_hash = digest(env.state_to_json(state))
    for line in lines[1:]:
        if line["type"] == "decision":
            try:
                state, _ = env.apply(state, line["seat"], line["applied"])
            except IllegalAction as exc:
                problems.append(f"seq {line['seq']}: recorded move is illegal: {exc}")
                return problems
            last_hash = digest(env.state_to_json(state))
        elif line["type"] == "event" and line["state_hash"] != last_hash:
            problems.append(f"seq {line['seq']}: state hash mismatch")
        elif line["type"] == "footer" and line["final_state_hash"] != last_hash:
            problems.append("final state hash mismatch")
    if footer.get("status") == "complete" and footer.get("outcomes") != env.outcome(state):
        problems.append("outcomes differ from the replayed final state")
    return problems
