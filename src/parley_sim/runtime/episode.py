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

import json
from collections import deque
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, field
from typing import Any

from parley_sim.core.hashing import canonical_json, digest
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
        if applied is not None and not _encodable(applied):
            # e.g. a lone surrogate from a JSON "\ud83d" escape: possibly a legal
            # move, but one that could never be hashed or written to the trace.
            strike, detail, applied = ("malformed_output",
                                       "action is not encodable as UTF-8 JSON", None)
        if applied is not None:
            try:
                self.env.apply(self.state, req.seat, applied)
            except IllegalAction as exc:
                strike, detail, applied = "illegal_action", str(exc), None
        if applied is None:
            applied = self.env.fallback_action(self.state, req.seat)
        self._commit(req.seat, req.observation, req.context, source="external",
                     submitted=_loggable(result.action), applied=applied,
                     reasoning=_loggable(result.reasoning),
                     explanation=_loggable(result.explanation), strike=strike,
                     detail=_loggable(detail), raw=_loggable(result.raw))

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
        final_state: JSON | None
        final_hash: str | None
        if status == "complete":
            outcomes = self.env.outcome(self.state)
            final_state = self.env.state_to_json(self.state)
            final_hash = digest(final_state)
        else:
            # The errored footer is always written, even when the state itself
            # is what failed, so a missing footer only ever means a cut file.
            outcomes = {}
            try:
                final_state = self.env.state_to_json(self.state)
                final_hash = digest(final_state)
            except Exception as exc:
                final_state, final_hash = None, None
                error = f"{error}; final state could not be serialized: {exc!r}"
        footer = {"type": "footer", "status": status, "outcomes": outcomes,
                  "n_decisions": self._n_decisions, "n_strikes": self._n_strikes,
                  "final_state": final_state, "final_state_hash": final_hash}
        if error is not None:
            footer["error"] = _loggable(error)
        self._emit(footer)
        # Set only once the footer is out: if building or writing a complete
        # footer fails, the driver's abort() still writes an errored one.
        self.result = EpisodeComplete(status=status, outcomes=outcomes,
                                      n_decisions=self._n_decisions, n_strikes=self._n_strikes)
        return self.result

    def _emit(self, line: dict[str, Any]) -> None:
        line = {"seq": self._seq, **line}
        self._seq += 1
        self.lines.append(line)
        if self._sink is not None:
            self._sink(line)


def _encodable(obj: Any) -> bool:
    """True if `obj` can be written to a trace: JSON-serializable, valid UTF-8."""
    try:
        json.dumps(obj, ensure_ascii=False).encode("utf-8")
    except (TypeError, ValueError):  # UnicodeEncodeError is a ValueError
        return False
    return True


def _utf8_safe(obj: Any) -> Any:
    if isinstance(obj, str):
        return obj.encode("utf-8", "backslashreplace").decode("utf-8")
    if isinstance(obj, dict):
        return {_utf8_safe(k): _utf8_safe(v) for k, v in obj.items()}
    if isinstance(obj, list | tuple):
        return [_utf8_safe(v) for v in obj]
    return obj


def _loggable(obj: Any) -> Any:
    """`obj` unchanged when it can be written to a trace; otherwise a JSON-safe
    copy with characters UTF-8 cannot encode backslash-escaped."""
    if _encodable(obj):
        return obj
    try:
        obj = json.loads(json.dumps(obj, default=repr, ensure_ascii=False))
    except ValueError:
        return repr(obj)
    return _utf8_safe(obj)


def _canon(obj: Any) -> Any:
    """A value as the trace stores it (tuples as lists, keys as strings), so a
    live value compares equal to its JSON-loaded record."""
    return json.loads(canonical_json(obj))


def verify(env: Environment, lines: list[dict[str, Any]],
           make_policy: Callable[[str, Seat], Policy] | None = None) -> list[str]:
    """Re-fold a recorded trace through the pure environment. Returns a list of
    mismatches (empty means the trace is exactly reproducible).

    Checked: the header's scenario, ground truth, seats and focal seat against
    the environment; for every decision, that its seat was the one to act and
    its turn, label, source, observation and decision context are what the
    replay gives; that the events after it are exactly the ones its move
    produces, in order; every state hash; and the footer's status, outcomes,
    final state and counts. With `make_policy`, scripted seats are rebuilt from
    the header and must reproduce every scripted move."""
    if not all(isinstance(l, dict) for l in lines):
        return ["malformed trace: every line must be a JSON object"]
    if not lines or lines[0].get("type") != "header":
        return ["no header"]
    if lines[-1].get("type") != "footer":
        return ["no footer: the trace is truncated"]
    problems: list[str] = []
    try:
        _verify(env, lines, make_policy, problems)
    except (KeyError, TypeError, AttributeError, ValueError, IndexError) as exc:
        problems.append(f"malformed trace: {exc!r}")
    return problems


def _verify_header(env: Environment, header: dict[str, Any],
                   make_policy: Callable[[str, Seat], Policy] | None,
                   problems: list[str]) -> tuple[Any, dict[Seat, Policy]]:
    spec = header["spec"]
    if header.get("env", {}).get("version") not in (None, env.version):
        problems.append(f"env version {header['env']['version']} != {env.version}")
    instance = env.generate(spec["seed"], spec["config"])
    if _canon(env.instance_to_json(instance)) != _canon(header["instance"]):
        problems.append("instance differs from header")
    if _canon(env.ground_truth(instance)) != _canon(header["ground_truth"]):
        problems.append("ground truth differs from header")
    labels: dict[str, str] = spec["seats"]
    seats = list(env.seats(instance))
    if sorted(labels) != sorted(seats):
        problems.append(f"header seats {sorted(labels)} != env seats {sorted(seats)}")
    if spec.get("focal_seat") is not None and spec["focal_seat"] not in seats:
        problems.append(f"focal seat {spec['focal_seat']!r} is not one of {seats}")
    policies: dict[str, dict[str, Any]] = header["policies"]
    for seat in sorted(set(policies) - set(seats)):
        problems.append(f"header has a policy for unknown seat {seat!r}")
    for seat, label in sorted(labels.items()):
        if label.startswith("policy:") and seat not in policies:
            problems.append(f"seat {seat!r} is {label!r} but the header has no policy for it")
    live: dict[Seat, Policy] = {}
    for seat, recorded in sorted(policies.items()):
        label = labels.get(seat, "")
        if label.startswith("policy:") and label.split(":", 1)[1] != recorded["name"]:
            problems.append(f"seat {seat!r} is {label!r} but its policy is {recorded['name']!r}")
        if make_policy is None or seat not in seats:
            continue
        try:
            policy = make_policy(recorded["name"], seat)
        except ValueError as exc:
            problems.append(f"policy {recorded['name']!r} for seat {seat!r}: {exc}")
            continue
        if policy.name != recorded["name"] or _canon(policy.params) != _canon(recorded["params"]):
            problems.append(f"policy for seat {seat!r} differs from header")
        live[seat] = policy
    return instance, live


def _verify(env: Environment, lines: list[dict[str, Any]],
            make_policy: Callable[[str, Seat], Policy] | None, problems: list[str]) -> None:
    header, footer = lines[0], lines[-1]
    instance, live = _verify_header(env, header, make_policy, problems)
    labels: dict[str, str] = header["spec"]["seats"]
    policies: dict[str, dict[str, Any]] = header["policies"]
    bad_seq = next((i for i, l in enumerate(lines) if l.get("seq") != i), None)
    if bad_seq is not None:
        problems.append(f"line {bad_seq + 1}: seq is {lines[bad_seq].get('seq')!r}, not {bad_seq}")

    state = env.initial_state(instance)
    last_hash = digest(env.state_to_json(state))
    if header.get("initial_state_hash") != last_hash:
        problems.append("initial state hash differs from header")
    decisions = [l for l in lines if l["type"] == "decision"]
    if footer.get("n_decisions") != len(decisions):
        problems.append("footer n_decisions differs from the decisions recorded")
    if footer.get("n_strikes") != sum(l.get("strike") is not None for l in decisions):
        problems.append("footer n_strikes differs from the strikes recorded")

    expected: deque[dict[str, Any]] = deque()  # events the last move produced, not yet seen
    for line in lines[1:-1]:
        seq, kind = line.get("seq"), line["type"]
        if kind == "event":
            if not expected:
                problems.append(f"seq {seq}: event the replay did not produce")
                continue
            want = expected.popleft()
            got = {k: v for k, v in line.items() if k != "seq"}
            if got.get("state_hash") != want["state_hash"]:
                problems.append(f"seq {seq}: state hash mismatch")
            elif _canon(got) != _canon(want):
                problems.append(f"seq {seq}: event differs from the replayed one")
            continue
        if expected:
            problems.append(f"seq {seq}: {len(expected)} event(s) missing before this line")
            expected.clear()
        if kind != "decision":
            problems.append(f"seq {seq}: unexpected {kind!r} line")
            continue
        seat = line["seat"]
        to_act = env.to_act(state)
        if seat != to_act:
            problems.append(f"seq {seq}: decision by {seat!r}, but the replay has "
                            + ("the episode over" if to_act is None else f"{to_act!r} to act"))
            return
        _verify_decision(env, state, line, labels, policies, live, problems)
        turn = env.turn(state)
        try:
            state, events = env.apply(state, seat, line["applied"])
        except IllegalAction as exc:
            problems.append(f"seq {seq}: recorded move is illegal: {exc}")
            return
        last_hash = digest(env.state_to_json(state))
        expected.extend({"type": "event", "turn": turn, "actor": seat, **ev,
                         "state_hash": last_hash} for ev in events)
    if expected:
        problems.append(f"{len(expected)} event(s) missing before the footer")
    _verify_footer(env, state, last_hash, footer, problems)


def _verify_decision(env: Environment, state: Any, line: dict[str, Any],
                     labels: dict[str, str], policies: dict[str, Any],
                     live: dict[Seat, Policy], problems: list[str]) -> None:
    seq, seat = line.get("seq"), line["seat"]
    if line["turn"] != env.turn(state):
        problems.append(f"seq {seq}: turn {line['turn']!r}, the replay is at {env.turn(state)}")
    if line["label"] != labels.get(seat):
        problems.append(f"seq {seq}: label {line['label']!r} differs from header "
                        f"{labels.get(seat)!r}")
    source = "scripted" if seat in policies else "external"
    if line["source"] != source:
        problems.append(f"seq {seq}: source {line['source']!r}, expected {source!r}")
    observation = env.observe(state, seat)
    if _canon(line["observation"]) != _canon(observation):
        problems.append(f"seq {seq}: observation differs from the replayed one")
    if _canon(line["context"]) != _canon(env.decision_context(state, seat)):
        problems.append(f"seq {seq}: decision context differs from the replayed one")
    applied, submitted = _canon(line["applied"]), _canon(line["submitted"])
    if source == "scripted":
        if line["strike"] is not None or submitted != applied:
            problems.append(f"seq {seq}: scripted decision with a strike or a changed move")
        if seat in live and _canon(live[seat].decide(observation)) != applied:
            problems.append(f"seq {seq}: policy {policies[seat]['name']!r} makes a different move")
    elif line["strike"] is None:
        if submitted != applied:
            problems.append(f"seq {seq}: applied move differs from the submitted one, "
                            "with no strike")
    elif applied not in (submitted, _canon(env.fallback_action(state, seat))):
        problems.append(f"seq {seq}: struck decision applied neither its move nor the fallback")


def _verify_footer(env: Environment, state: Any, last_hash: str, footer: dict[str, Any],
                   problems: list[str]) -> None:
    status = footer.get("status")
    if status == "complete":
        if env.to_act(state) is not None:
            problems.append(f"footer says complete, but {env.to_act(state)!r} is still to act")
        if _canon(footer.get("outcomes")) != _canon(env.outcome(state)):
            problems.append("outcomes differ from the replayed final state")
    elif status == "errored":
        if footer.get("outcomes"):
            problems.append("errored footer records outcomes")
        if footer.get("final_state") is None and footer.get("final_state_hash") is None:
            return  # the state itself could not be serialized when the episode failed
    else:
        problems.append(f"unknown footer status {status!r}")
    if footer.get("final_state_hash") != last_hash:
        problems.append("final state hash mismatch")
    if _canon(footer.get("final_state")) != _canon(env.state_to_json(state)):
        problems.append("final state differs from the replayed final state")
