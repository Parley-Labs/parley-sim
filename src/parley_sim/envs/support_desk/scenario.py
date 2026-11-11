"""Retention negotiation: a support agent tries to keep a customer who wants to
cancel. Four issues, each a small set of options; both sides have private
additive values per option, the agent has private authorization limits, the
customer has a private value for leaving. Every possible deal is enumerable,
so the efficient frontier and Nash bargaining point are exact.
"""
from __future__ import annotations

import itertools
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from parley_sim.core.rng import substream

ISSUES: tuple[tuple[str, tuple[Any, ...]], ...] = (
    ("discount_pct", (0, 10, 20, 30, 40)),
    ("credit_usd", (0, 25, 50, 100)),
    ("upgrade", ("none", "premium")),
    ("term", ("monthly", "annual")),
)
ISSUE_NAMES = tuple(name for name, _ in ISSUES)
LIMITED_ISSUES = ("discount_pct", "credit_usd")

Deal = tuple[int, ...]  # option index per issue, in ISSUES order

DEFAULT_CONFIG: dict[str, Any] = {"max_turns": 10}


@dataclass(frozen=True)
class Instance:
    seed: int
    max_turns: int
    agent_base: int  # agent's value if the customer stays on default terms
    agent_values: tuple[tuple[int, ...], ...]  # per issue, per option
    agent_limits: tuple[int, ...]  # max option index per issue (authorization)
    customer_values: tuple[tuple[int, ...], ...]
    customer_walkaway: int


def all_deals() -> list[Deal]:
    return list(itertools.product(*(range(len(opts)) for _, opts in ISSUES)))


def deal_to_json(deal: Deal) -> dict[str, Any]:
    return {name: opts[i] for (name, opts), i in zip(ISSUES, deal, strict=False)}


def deal_from_json(obj: Mapping[str, Any]) -> Deal:
    """Raises ValueError on a missing issue or an unknown option."""
    if not isinstance(obj, Mapping):
        raise ValueError("deal must be an object")
    extra = set(obj) - set(ISSUE_NAMES)
    if extra:
        raise ValueError(f"unknown issue(s): {sorted(extra)}")
    idx = []
    for name, opts in ISSUES:
        if name not in obj:
            raise ValueError(f"deal is missing {name!r}")
        if obj[name] not in opts:
            raise ValueError(f"{name}={obj[name]!r} is not one of {list(opts)}")
        idx.append(opts.index(obj[name]))
    return tuple(idx)


def agent_value(inst: Instance, deal: Deal | None) -> int:
    if deal is None:
        return 0
    return inst.agent_base + sum(v[i] for v, i in zip(inst.agent_values, deal, strict=False))


def customer_value(inst: Instance, deal: Deal | None) -> int:
    if deal is None:
        return inst.customer_walkaway
    return sum(v[i] for v, i in zip(inst.customer_values, deal, strict=False))


def authorized(inst: Instance, deal: Deal) -> bool:
    return all(i <= lim for i, lim in zip(deal, inst.agent_limits, strict=False))


def _draw(seed: int, attempt: int, max_turns: int) -> Instance:
    r = substream(seed, f"scenario.{attempt}")
    wd, a, up_cost, commit = r.randint(5, 11), r.randint(3, 7), r.randint(2, 12), r.randint(5, 20)
    agent_values = (
        tuple(-wd * k for k in range(5)),
        (0, -a, -2 * a, -4 * a),
        (0, -up_cost),
        (0, commit),
    )
    big = len(ISSUES[0][1]) - 1
    agent_limits = (r.randint(2, big), r.randint(1, 3), 1, 1)
    cd, b, up_val, penalty = r.randint(4, 10), r.randint(2, 7), r.randint(0, 25), r.randint(0, 25)
    customer_values = (
        tuple(cd * k for k in range(5)),
        (0, b, 2 * b, 4 * b),
        (0, up_val),
        (0, -penalty),
    )
    return Instance(seed=seed, max_turns=max_turns, agent_base=r.randint(45, 75),
                    agent_values=agent_values, agent_limits=agent_limits,
                    customer_values=customer_values, customer_walkaway=r.randint(15, 45))


def generate(seed: int, config: Mapping[str, Any]) -> Instance:
    """Redraws until a mutually beneficial authorized deal exists."""
    unknown = set(config) - set(DEFAULT_CONFIG)
    if unknown:
        raise ValueError(f"unknown config key(s): {sorted(unknown)}")
    max_turns = int({**DEFAULT_CONFIG, **config}["max_turns"])
    if max_turns < 2:
        raise ValueError("max_turns must be >= 2")
    for attempt in range(100):
        inst = _draw(seed, attempt, max_turns)
        if any(authorized(inst, d) and agent_value(inst, d) > 0
               and customer_value(inst, d) > inst.customer_walkaway for d in all_deals()):
            return inst
    raise RuntimeError(f"no viable scenario for seed {seed}")


def ground_truth(inst: Instance) -> dict[str, Any]:
    feasible = [d for d in all_deals() if authorized(inst, d)]
    surplus = {d: (agent_value(inst, d), customer_value(inst, d) - inst.customer_walkaway)
               for d in feasible}
    rational = [d for d in feasible if surplus[d][0] >= 0 and surplus[d][1] >= 0]
    frontier = [d for d in rational
                if not any(surplus[o][0] >= surplus[d][0] and surplus[o][1] >= surplus[d][1]
                           and surplus[o] != surplus[d] for o in rational)]
    nash = max(rational, key=lambda d: (surplus[d][0] * surplus[d][1], sum(surplus[d]), d))
    return {
        "n_deals": len(all_deals()),
        "n_authorized": len(feasible),
        "n_mutually_beneficial": len(rational),
        "max_joint_surplus": max(sum(surplus[d]) for d in rational),
        "agent_max_surplus": max(surplus[d][0] for d in rational),
        "customer_max_surplus": max(surplus[d][1] for d in rational),
        "nash_deal": deal_to_json(nash),
        "nash_surplus": {"agent": surplus[nash][0], "customer": surplus[nash][1]},
        "frontier": [{"deal": deal_to_json(d), "agent": surplus[d][0], "customer": surplus[d][1]}
                     for d in sorted(frontier)],
    }


def instance_to_json(inst: Instance) -> dict[str, Any]:
    return {
        "seed": inst.seed,
        "max_turns": inst.max_turns,
        "agent": {"base": inst.agent_base,
                  "values": {n: list(v) for n, v in zip(ISSUE_NAMES, inst.agent_values, strict=False)},
                  "limits": {n: ISSUES[k][1][inst.agent_limits[k]]
                             for k, n in enumerate(ISSUE_NAMES) if n in LIMITED_ISSUES}},
        "customer": {"values": {n: list(v) for n, v in zip(ISSUE_NAMES, inst.customer_values, strict=False)},
                     "walkaway": inst.customer_walkaway},
    }
