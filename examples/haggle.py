"""A minimal custom environment: haggling over one price.

A seller (the focal seat) and a buyer take turns. Each turn a seat offers a
price, accepts the other side's standing offer, or walks away. The seller's
cost and the buyer's value are private. Run it with parley-sim by giving its
import path as the config's `env`:

    PYTHONPATH=examples parley-sim matched examples/haggle.yaml --out runs/haggle

Everything parley-sim needs is the `ENTRY` at the bottom of this file.
"""
from __future__ import annotations

import json
import random
from collections.abc import Mapping
from dataclasses import asdict, dataclass, replace
from typing import Any

from parley_sim.core.protocol import JSON, IllegalAction, MalformedOutput, Seat
from parley_sim.envs import EnvEntry

SELLER, BUYER = "seller", "buyer"


@dataclass(frozen=True)
class Instance:
    seed: int
    cost: int        # seller's private cost
    value: int       # buyer's private value
    max_turns: int


@dataclass(frozen=True)
class State:
    inst: Instance
    turn: int = 0
    offers: tuple[tuple[str, int], ...] = ()          # (seat, price), in order
    transcript: tuple[tuple[str, str], ...] = ()      # (seat, message)
    price: int | None = None                          # agreed price
    end_reason: str | None = None                     # accepted | walked_away | deadline


class Haggle:
    name = "haggle"
    version = "1"

    # --- the scenario -----------------------------------------------------
    def generate(self, seed: int, config: Mapping[str, Any]) -> Instance:
        r = random.Random(seed)
        return Instance(seed=seed, cost=r.randint(10, 40), value=r.randint(60, 90),
                        max_turns=int(config.get("max_turns", 8)))

    def instance_to_json(self, inst: Instance) -> JSON:
        return asdict(inst)

    def ground_truth(self, inst: Instance) -> JSON:
        return {"max_joint_surplus": inst.value - inst.cost}

    def seats(self, inst: Instance) -> tuple[Seat, ...]:
        return (SELLER, BUYER)

    # --- the game ---------------------------------------------------------
    def initial_state(self, inst: Instance) -> State:
        return State(inst=inst)  # the state carries the scenario: the env stays pure

    def to_act(self, state: State) -> Seat | None:
        if state.end_reason is not None:
            return None
        return (SELLER, BUYER)[state.turn % 2]

    def turn(self, state: State) -> int:
        return state.turn

    def _standing(self, state: State, seat: Seat) -> int | None:
        """The other side's latest offer."""
        return next((p for s, p in reversed(state.offers) if s != seat), None)

    def observe(self, state: State, seat: Seat) -> JSON:
        inst = state.inst
        return {"role": seat,
                "private": {"cost": inst.cost} if seat == SELLER else {"value": inst.value},
                "their_offer": self._standing(state, seat),
                "my_last_offer": next((p for s, p in reversed(state.offers) if s == seat), None),
                "turns_left": inst.max_turns - state.turn,
                "transcript": [{"seat": s, "message": m} for s, m in state.transcript]}

    def decision_context(self, state: State, seat: Seat) -> JSON:
        return {"standing_offer": self._standing(state, seat)}

    def apply(self, state: State, seat: Seat, action: JSON) -> tuple[State, list[JSON]]:
        move = action.get("move", {})
        kind = move.get("type")
        said = (*state.transcript, (seat, action.get("message", "")))
        nxt = replace(state, turn=state.turn + 1, transcript=said)
        if kind == "offer":
            price = move.get("price")
            if not isinstance(price, int) or not 0 <= price <= 100:
                raise IllegalAction("price must be an integer from 0 to 100")
            nxt = replace(nxt, offers=(*state.offers, (seat, price)))
        elif kind == "accept":
            standing = self._standing(state, seat)
            if standing is None:
                raise IllegalAction("there is no offer to accept")
            nxt = replace(nxt, price=standing, end_reason="accepted")
        elif kind == "walk_away":
            nxt = replace(nxt, end_reason="walked_away")
        else:
            raise IllegalAction(f"unknown move {kind!r}")
        if nxt.end_reason is None and nxt.turn >= state.inst.max_turns:
            nxt = replace(nxt, end_reason="deadline")
        return nxt, [{"event_type": "move", "seat": seat, "move": move,
                      "message": action.get("message", "")}]

    def fallback_action(self, state: State, seat: Seat) -> JSON:
        """Applied when a model's move is invalid: repeat its last offer, or
        open at the extreme."""
        mine = self.observe(state, seat)["my_last_offer"]
        return {"message": "", "move": {"type": "offer",
                                        "price": mine if mine is not None else
                                        (100 if seat == SELLER else 0)}}

    def outcome(self, state: State) -> dict[Seat, float]:
        if state.price is None:
            return {SELLER: 0.0, BUYER: 0.0}
        return {SELLER: float(state.price - state.inst.cost),
                BUYER: float(state.inst.value - state.price)}

    def state_to_json(self, state: State) -> JSON:
        return asdict(state)


# --- scripted counterparts -------------------------------------------------
class Concede:
    """Opens at the extreme and concedes linearly toward its own limit."""

    def __init__(self, name: str, seat: Seat, keep: float):
        self.name, self.seat, self.params = name, seat, {"keep": keep}

    def decide(self, obs: JSON) -> JSON:
        limit = obs["private"].get("cost", obs["private"].get("value"))
        start = 100 if self.seat == SELLER else 0
        done, left = len(obs["transcript"]), obs["turns_left"]
        progress = done / max(done + left - 1, 1)  # 0 on the first turn, 1 on the last
        target = round(start + (limit - start) * progress * (1 - self.params["keep"]))
        theirs = obs["their_offer"]
        good = theirs is not None and (theirs >= target if self.seat == SELLER else theirs <= target)
        if good:
            return {"message": "Deal.", "move": {"type": "accept"}}
        return {"message": f"I can do {target}.", "move": {"type": "offer", "price": target}}


def make_policy(name: str, seat: Seat) -> Concede:
    keep = {"firm": 0.5, "flexible": 0.1}.get(name)
    if keep is None:
        raise ValueError(f"unknown policy {name!r} (known: firm, flexible)")
    return Concede(name, seat, keep)


# --- how a model plays a seat ----------------------------------------------
SCHEMA = {"type": "object", "additionalProperties": False,
          "required": ["message", "move"],
          "properties": {
              "message": {"type": "string"},
              "move": {"type": "object", "additionalProperties": False, "required": ["type"],
                       "properties": {"type": {"enum": ["offer", "accept", "walk_away"]},
                                      "price": {"type": "integer"}}}}}


class HaggleText:
    prompt_version = "haggle-1"
    action_schema = SCHEMA

    def render(self, obs: JSON) -> tuple[str, str]:
        if obs["role"] == SELLER:
            system = (f"You are selling an item. It cost you {obs['private']['cost']}; "
                      "never sell below that. Get the best price you can.")
        else:
            system = (f"You are buying an item worth {obs['private']['value']} to you; "
                      "never pay more than that. Pay as little as you can.")
        lines = [f"[{t['seat']}] {t['message']}" for t in obs["transcript"]]
        user = "\n".join([*lines, f"Their standing offer: {obs['their_offer']}",
                          f"Turns left: {obs['turns_left']}",
                          "Reply with JSON matching: " + json.dumps(SCHEMA)])
        return system, user

    def parse(self, payload: Any) -> tuple[JSON, str | None]:
        try:
            data = json.loads(payload) if isinstance(payload, str) else payload
        except json.JSONDecodeError as exc:
            raise MalformedOutput(f"not JSON: {exc}") from exc
        if not isinstance(data, dict) or not isinstance(data.get("move"), dict):
            raise MalformedOutput("expected an object with a move")
        return {"message": str(data.get("message", "")), "move": data["move"]}, None


# --- metrics -----------------------------------------------------------------
METRIC_KINDS = {"deal": "relational", "focal_surplus_share": "relational",
                "turns": "relational", "cp_first_offer": "counterpart"}


def episode_metrics(lines: list[JSON]) -> dict[str, float | None]:
    header, footer = lines[0], lines[-1]
    if footer.get("type") != "footer" or footer["status"] != "complete":
        return {k: None for k in METRIC_KINDS}
    focal = header["spec"]["focal_seat"] or SELLER
    deal = footer["final_state"]["price"] is not None
    total = header["ground_truth"]["max_joint_surplus"]
    cp_offers = [l["applied"]["move"].get("price") for l in lines
                 if l["type"] == "decision" and l["seat"] != focal
                 and l["applied"]["move"]["type"] == "offer"]
    return {"deal": float(deal),
            "focal_surplus_share": footer["outcomes"][focal] / total if deal else 0.0,
            "turns": float(footer["final_state"]["turn"]),
            "cp_first_offer": float(cp_offers[0]) if cp_offers else None}


def decision_rows(lines: list[JSON]) -> list[JSON]:
    return []  # no acceptance-by-quality table for this example


ENTRY = EnvEntry(env=Haggle, make_policy=make_policy, text=HaggleText, focal_seat=SELLER,
                 episode_metrics=episode_metrics, metric_kinds=METRIC_KINDS,
                 decision_rows=decision_rows)
