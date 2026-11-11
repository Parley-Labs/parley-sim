from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

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


_registered: dict[str, EnvEntry] = {}


def _builtin() -> dict[str, EnvEntry]:
    from parley_sim.envs import support_desk as sd
    from parley_sim.envs.support_desk import metrics as sdm
    return {"support_desk": EnvEntry(sd.SupportDesk, sd.make_policy, sd.SupportDeskText,
                                     sd.AGENT, sdm.episode_metrics, sdm.METRIC_KINDS,
                                     sdm.decision_rows, sdm.claim_rows,
                                     sdm.counterpart_opening)}


def register(name: str, entry: EnvEntry) -> None:
    """Make an environment available under `name` in this process."""
    if not isinstance(entry, EnvEntry):
        raise TypeError(f"{name!r}: expected an EnvEntry, got {type(entry).__name__}")
    _registered[name] = entry


def registry() -> dict[str, EnvEntry]:
    return {**_builtin(), **_registered}


def _load(target: str) -> EnvEntry:
    import importlib
    module, _, attr = target.partition(":")
    obj = getattr(importlib.import_module(module), attr)
    entry = obj() if callable(obj) and not isinstance(obj, EnvEntry) else obj
    if not isinstance(entry, EnvEntry):
        raise TypeError(f"{target!r} is not an EnvEntry (or a function returning one)")
    return entry


def get(name: str) -> EnvEntry:
    """An environment by name: built in, registered with `register`, given as
    an import path ("my_pkg.envs:ENTRY"), or published by an installed package
    under the "parley_sim.envs" entry-point group."""
    envs = registry()
    if name in envs:
        return envs[name]
    if ":" in name:
        entry = _load(name)
        register(name, entry)
        register(entry.env().name, entry)  # traces record the env's own name
        return entry
    from importlib.metadata import entry_points
    for ep in entry_points(group="parley_sim.envs"):
        if ep.name == name:
            entry = _load(ep.value)
            register(name, entry)
            return entry
    raise ValueError(f"unknown env {name!r} (known: {sorted(envs)}; or give an import "
                     f"path like 'my_pkg.envs:ENTRY')")
