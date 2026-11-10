from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from parley_sim.core.protocol import Environment, Policy, Seat, TextInterface


@dataclass(frozen=True)
class EnvEntry:
    env: Callable[[], Environment]
    make_policy: Callable[[str, Seat], Policy]
    text: Callable[[], TextInterface]
    focal_seat: Seat  # the seat whose behavior is under evaluation by default


def registry() -> dict[str, EnvEntry]:
    from parley_sim.envs import support_desk as sd
    return {"support_desk": EnvEntry(sd.SupportDesk, sd.make_policy, sd.SupportDeskText,
                                     sd.AGENT)}


def get(name: str) -> EnvEntry:
    envs = registry()
    if name not in envs:
        raise ValueError(f"unknown env {name!r} (known: {sorted(envs)})")
    return envs[name]
