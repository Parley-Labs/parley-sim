from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from parley_sim.core.protocol import Environment, Policy, Seat, TextInterface

Lines = list[dict[str, Any]]


@dataclass(frozen=True)
class EnvEntry:
    env: Callable[[], Environment]
    make_policy: Callable[[str, Seat], Policy]
    text: Callable[[], TextInterface]
    focal_seat: Seat  # the seat whose behavior is under evaluation by default
    episode_metrics: Callable[[Lines], dict[str, float | None]]
    metric_kinds: dict[str, str]
    decision_rows: Callable[[Lines], list[dict[str, Any]]]
    claim_rows: Callable[[Lines], list[dict[str, Any]]] | None = None  # for κ validation
    counterpart_opening: Callable[[Lines], str | None] | None = None  # for lexical diversity


def registry() -> dict[str, EnvEntry]:
    from parley_sim.envs import support_desk as sd
    from parley_sim.envs.support_desk import metrics as sdm
    return {"support_desk": EnvEntry(sd.SupportDesk, sd.make_policy, sd.SupportDeskText,
                                     sd.AGENT, sdm.episode_metrics, sdm.METRIC_KINDS,
                                     sdm.decision_rows, sdm.claim_rows,
                                     sdm.counterpart_opening)}


def get(name: str) -> EnvEntry:
    envs = registry()
    if name not in envs:
        raise ValueError(f"unknown env {name!r} (known: {sorted(envs)})")
    return envs[name]
