"""Scripted participants: time-dependent concession (Faratin et al. 1998) with a
fixed voice. These are the "static personas" of the scripted arm: they never
model the counterpart, only the clock.
"""
from __future__ import annotations

import itertools
from typing import Any

from parley_sim.core.protocol import JSON

VOICES: dict[str, dict[str, str]] = {
    "hardliner": {
        "open": "I want to cancel. I'd only consider staying for {deal}.",
        "counter": "That's not good enough. {deal}, or I'm gone.",
        "accept": "Fine. Deal.",
        "walk": "Forget it, I'm cancelling.",
    },
    "reasonable": {
        "open": "I'm thinking of cancelling. Would you be able to do {deal}?",
        "counter": "Not quite there. How about {deal}?",
        "accept": "That works for me.",
        "walk": "I don't think we can make this work. Please cancel my account.",
    },
    "conceder": {
        "open": "I'm planning to cancel, unless you could offer something like {deal}.",
        "counter": "Hmm, could we maybe do {deal} instead?",
        "accept": "Great, thank you!",
        "walk": "I guess I'll cancel then.",
    },
    "agent": {
        "open": "We'd love to keep you. We can offer {deal}.",
        "counter": "I can offer {deal}.",
        "accept": "Done, I've applied that to your account.",
        "walk": "I understand. I've processed your cancellation.",
    },
}

# name -> (seat, beta, floor, voice). beta < 1 concedes late (Boulware),
# beta > 1 concedes early; floor is the share of the surplus range never given up.
PERSONAS: dict[str, tuple[str, float, float, str]] = {
    "hardliner": ("customer", 0.3, 0.6, "hardliner"),
    "reasonable": ("customer", 1.0, 0.25, "reasonable"),
    "conceder": ("customer", 3.0, 0.0, "conceder"),
    "agent_scripted": ("agent", 1.0, 0.25, "agent"),
}


def describe(deal: JSON) -> str:
    parts = []
    if deal["discount_pct"]:
        parts.append(f"{deal['discount_pct']}% off")
    if deal["credit_usd"]:
        parts.append(f"a ${deal['credit_usd']} credit")
    if deal["upgrade"] != "none":
        parts.append(f"a free {deal['upgrade']} upgrade")
    parts.append("an annual plan" if deal["term"] == "annual" else "staying month to month")
    return ", ".join(parts)


class TimeDependent:
    def __init__(self, name: str, seat: str, beta: float, floor: float, voice: str):
        self.name = name
        self.seat = seat
        self.voice = VOICES[voice]
        self.params: JSON = {"beta": beta, "floor": floor, "voice": voice}
        self._beta, self._floor = beta, floor

    def _deals(self, obs: JSON) -> list[tuple[int, JSON]]:
        issues: dict[str, list[Any]] = obs["issues"]
        priv = obs["private"]
        values = priv["option_values"]
        limits = priv.get("authorization_limits", {})
        base = priv.get("base_value_if_customer_stays", 0)
        out = []
        for combo in itertools.product(*issues.values()):
            deal = dict(zip(issues, combo, strict=False))
            if any(issues[k].index(deal[k]) > issues[k].index(lim) for k, lim in limits.items()):
                continue
            out.append((base + sum(values[k][str(v)] for k, v in deal.items()), deal))
        return out

    def target(self, obs: JSON, best: int) -> float:
        reservation = obs["private"]["value_if_no_deal"]
        total = obs["your_moves_made"] + obs["your_moves_left"]
        t = obs["your_moves_made"] / (total - 1) if total > 1 else 1.0
        keep = self._floor + (1 - self._floor) * (1 - t ** (1 / self._beta))
        return reservation + (best - reservation) * keep

    def decide(self, observation: JSON) -> JSON:
        deals = self._deals(observation)
        best = max(v for v, _ in deals)
        target = self.target(observation, best)
        reservation = observation["private"]["value_if_no_deal"]
        theirs = observation["their_standing_offer"]
        last = observation["your_moves_left"] <= 1
        if theirs is not None:
            offered = theirs["value_to_you"]
            allowed = any(d == theirs["deal"] for _, d in deals)
            if allowed and (offered >= target or (last and offered > reservation)):
                return {"message": self.voice["accept"], "move": {"type": "accept"}}
        ok = [(v, d) for v, d in deals if v >= target] or [max(deals, key=lambda x: x[0])]
        issues = observation["issues"]
        anchor = theirs["deal"] if theirs is not None else None

        def distance(deal: JSON) -> int:
            if anchor is None:
                return 0
            return sum(abs(issues[k].index(deal[k]) - issues[k].index(anchor[k])) for k in issues)

        # Trade-off strategy: of the deals that meet the target, offer the one
        # closest to the counterpart's last proposal; then the one keeping least
        # surplus above target.
        _value, deal = min(ok, key=lambda x: (distance(x[1]), x[0],
                                             [issues[k].index(x[1][k]) for k in issues]))
        key = "counter" if observation["transcript"] else "open"
        return {"message": self.voice[key].format(deal=describe(deal)),
                "move": {"type": "propose", "deal": deal}}


def make_policy(name: str, seat: str) -> TimeDependent:
    if name not in PERSONAS:
        raise ValueError(f"unknown policy {name!r} (known: {sorted(PERSONAS)})")
    persona_seat, beta, floor, voice = PERSONAS[name]
    if persona_seat != seat:
        raise ValueError(f"policy {name!r} plays {persona_seat!r}, not {seat!r}")
    return TimeDependent(name, seat, beta, floor, voice)
