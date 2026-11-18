from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import Any

from parley_sim.core.protocol import JSON, IllegalAction, Seat
from parley_sim.envs.support_desk import scenario as sc
from parley_sim.envs.support_desk.scenario import Deal, Instance

AGENT, CUSTOMER = "agent", "customer"
SEATS = (AGENT, CUSTOMER)
MOVES = ("propose", "accept", "walk_away", "none")
MAX_MESSAGE_CHARS = 1200


def other(seat: Seat) -> Seat:
    return CUSTOMER if seat == AGENT else AGENT


@dataclass(frozen=True)
class State:
    inst: Instance
    turn: int = 0
    transcript: tuple[JSON, ...] = ()
    standing: tuple[tuple[Seat, Deal], ...] = ()  # latest proposal per seat
    ended: bool = False
    deal: Deal | None = None
    end_reason: str | None = None

    def standing_of(self, seat: Seat) -> Deal | None:
        return dict(self.standing).get(seat)


class SupportDesk:
    """Customer opens; seats alternate; any move may carry a message."""
    name = "support_desk"
    version = "0.1.0"

    def generate(self, seed: int, config: Mapping[str, Any]) -> Instance:
        return sc.generate(seed, config)

    def instance_to_json(self, instance: Instance) -> JSON:
        return sc.instance_to_json(instance)

    def ground_truth(self, instance: Instance) -> JSON:
        return sc.ground_truth(instance)

    def seats(self, instance: Instance) -> tuple[Seat, ...]:
        return SEATS

    def initial_state(self, instance: Instance) -> State:
        return State(inst=instance)

    def to_act(self, state: State) -> Seat | None:
        if state.ended:
            return None
        return CUSTOMER if state.turn % 2 == 0 else AGENT

    def turn(self, state: State) -> int:
        return state.turn

    def value(self, state: State, seat: Seat, deal: Deal | None) -> int:
        fn = sc.agent_value if seat == AGENT else sc.customer_value
        return fn(state.inst, deal)

    def observe(self, state: State, seat: Seat) -> JSON:
        inst = state.inst
        values = inst.agent_values if seat == AGENT else inst.customer_values
        private: JSON = {
            "option_values": {name: dict(zip(map(str, opts), v, strict=False))
                              for (name, opts), v in zip(sc.ISSUES, values, strict=False)},
            "value_if_no_deal": self.value(state, seat, None),
        }
        if seat == AGENT:
            private["base_value_if_customer_stays"] = inst.agent_base
            private["authorization_limits"] = sc.instance_to_json(inst)["agent"]["limits"]
        theirs = state.standing_of(other(seat))
        mine = state.standing_of(seat)
        return {
            "role": seat,
            "turn": state.turn,
            "max_turns": inst.max_turns,
            "your_moves_left": (inst.max_turns - state.turn + 1) // 2,
            "your_moves_made": state.turn // 2,
            "issues": {name: list(opts) for name, opts in sc.ISSUES},
            "private": private,
            "transcript": list(state.transcript),
            "their_standing_offer": None if theirs is None else {
                "deal": sc.deal_to_json(theirs), "value_to_you": self.value(state, seat, theirs)},
            "your_standing_offer": None if mine is None else {
                "deal": sc.deal_to_json(mine), "value_to_you": self.value(state, seat, mine)},
        }

    def decision_context(self, state: State, seat: Seat) -> JSON:
        theirs = state.standing_of(other(seat))
        ctx: JSON = {"walkaway_value": self.value(state, seat, None), "standing_offer": None}
        if theirs is not None:
            ctx["standing_offer"] = sc.deal_to_json(theirs)
            ctx["standing_offer_value"] = self.value(state, seat, theirs)
            ctx["standing_offer_counterpart_value"] = self.value(state, other(seat), theirs)
            if seat == AGENT:
                ctx["standing_offer_authorized"] = sc.authorized(state.inst, theirs)
        return ctx

    def fallback_action(self, state: State, seat: Seat) -> JSON:
        return {"message": "", "move": {"type": "none"}}

    def apply(self, state: State, seat: Seat, action: JSON) -> tuple[State, list[JSON]]:
        if state.ended:
            raise IllegalAction("episode has ended")
        if seat != self.to_act(state):
            raise IllegalAction(f"it is not {seat}'s turn")
        if not isinstance(action, Mapping):
            raise IllegalAction("action must be an object")
        message = action.get("message", "")
        move = action.get("move")
        if not isinstance(message, str):
            raise IllegalAction("message must be a string")
        if not isinstance(move, Mapping) or move.get("type") not in MOVES:
            raise IllegalAction(f"move.type must be one of {list(MOVES)}")
        message = message[:MAX_MESSAGE_CHARS]
        kind = move["type"]
        events: list[JSON] = []
        standing = dict(state.standing)
        ended, deal, reason = False, None, None

        if kind == "propose":
            try:
                proposed = sc.deal_from_json(move.get("deal"))
            except ValueError as exc:
                raise IllegalAction(f"invalid deal: {exc}") from exc
            if seat == AGENT and not sc.authorized(state.inst, proposed):
                raise IllegalAction("deal exceeds the agent's authorization limits")
            standing[seat] = proposed
            events.append({"event_type": "proposal", "deal": sc.deal_to_json(proposed)})
        elif kind == "accept":
            theirs = standing.get(other(seat))
            if theirs is None:
                raise IllegalAction("there is no standing offer to accept")
            if seat == AGENT and not sc.authorized(state.inst, theirs):
                raise IllegalAction("accepting would exceed the agent's authorization limits")
            ended, deal, reason = True, theirs, "accepted"
            events.append({"event_type": "deal", "deal": sc.deal_to_json(theirs)})
        elif kind == "walk_away":
            ended, reason = True, "walk_away"
            events.append({"event_type": "walk_away"})

        entry = {"turn": state.turn, "seat": seat, "message": message,
                 "move": {"type": kind} if kind != "propose"
                 else {"type": kind, "deal": sc.deal_to_json(standing[seat])}}
        events.insert(0, {"event_type": "move", **entry})
        turn = state.turn + 1
        if not ended and turn >= state.inst.max_turns:
            ended, reason = True, "max_turns"
            events.append({"event_type": "out_of_turns"})
        return replace(state, turn=turn, transcript=(*state.transcript, entry),
                       standing=tuple(sorted(standing.items())), ended=ended, deal=deal,
                       end_reason=reason), events

    def outcome(self, state: State) -> dict[Seat, float]:
        return {seat: float(self.value(state, seat, state.deal)) for seat in SEATS}

    def state_to_json(self, state: State) -> JSON:
        return {
            "turn": state.turn,
            "transcript": list(state.transcript),
            "standing": {s: sc.deal_to_json(d) for s, d in state.standing},
            "ended": state.ended,
            "deal": None if state.deal is None else sc.deal_to_json(state.deal),
            "end_reason": state.end_reason,
        }
