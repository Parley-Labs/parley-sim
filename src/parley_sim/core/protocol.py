"""Contracts every environment, scripted policy and text interface implements.

Environments are pure and sequential: exactly one seat acts at a time
(`to_act`), `apply` never mutates its input, and `observe` is the only view of
state any seat ever receives. Actions and observations are plain JSON objects
so traces, model I/O and external harness adapters need no per-env codecs.
"""
from __future__ import annotations

from typing import Any, Mapping, Protocol, runtime_checkable

Seat = str
JSON = dict[str, Any]


class IllegalAction(ValueError):
    """An action is well-formed but not legal in the current state."""


class MalformedOutput(ValueError):
    """Model output could not be parsed into an action."""


@runtime_checkable
class Environment(Protocol):
    name: str
    version: str

    def generate(self, seed: int, config: Mapping[str, Any]) -> Any: ...
    def instance_to_json(self, instance: Any) -> JSON: ...
    def ground_truth(self, instance: Any) -> JSON: ...
    def seats(self, instance: Any) -> tuple[Seat, ...]: ...

    def initial_state(self, instance: Any) -> Any: ...
    def to_act(self, state: Any) -> Seat | None:
        """The seat that must act next, or None once the episode is over."""
    def turn(self, state: Any) -> int: ...
    def observe(self, state: Any, seat: Seat) -> JSON: ...
    def decision_context(self, state: Any, seat: Seat) -> JSON:
        """What this decision is about (e.g. the standing offer being
        answered), recorded with every decision so declines are as visible
        as acceptances."""
    def apply(self, state: Any, seat: Seat, action: JSON) -> tuple[Any, list[JSON]]:
        """Pure transition. Raises IllegalAction; returns (new_state, events)."""
    def fallback_action(self, state: Any, seat: Seat) -> JSON: ...
    def outcome(self, state: Any) -> dict[Seat, float]: ...
    def state_to_json(self, state: Any) -> JSON: ...


@runtime_checkable
class Policy(Protocol):
    """A scripted (non-adaptive-by-design) participant."""
    name: str
    params: JSON

    def decide(self, observation: JSON) -> JSON: ...


@runtime_checkable
class TextInterface(Protocol):
    """How a language model plays a seat of an environment."""
    prompt_version: str
    action_schema: JSON

    def render(self, observation: JSON) -> tuple[str, str]:
        """(system, user) prompt for this observation."""
    def parse(self, payload: JSON) -> tuple[JSON, str | None]:
        """(action, explanation): the move, plus the model's own short account of
        it if the interface asks for one. Raises MalformedOutput."""
