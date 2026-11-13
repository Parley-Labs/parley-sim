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


def is_import_path(name: str) -> bool:
    return ":" in name


def _load(target: str) -> EnvEntry:
    """Import `module:attr` and check it is an EnvEntry, or a function taking
    no arguments that returns one, before anything else uses it."""
    import importlib
    import inspect
    module, _, attr = target.partition(":")
    if not module or not attr:
        raise ValueError(f"{target!r} is not an import path like 'my_pkg.envs:ENTRY'")
    try:
        obj = getattr(importlib.import_module(module), attr)
    except AttributeError as exc:
        raise ValueError(f"{target!r}: module {module!r} has no attribute {attr!r}") from exc
    if isinstance(obj, EnvEntry):
        return obj
    if not callable(obj) or isinstance(obj, type):
        raise TypeError(f"{target!r} is not an EnvEntry (or a function returning one)")
    try:
        inspect.signature(obj).bind()
    except TypeError as exc:
        raise TypeError(f"{target!r} is a function that needs arguments; expected an EnvEntry "
                        "or a function taking none") from exc
    except ValueError:  # no signature available; let the call decide
        pass
    entry = obj()
    if not isinstance(entry, EnvEntry):
        raise TypeError(f"{target!r} returned {type(entry).__name__}, not an EnvEntry")
    return entry


def _bind_alias(alias: str, entry: EnvEntry, source: str) -> None:
    """Register `alias` for an entry loaded from `source`, unless the name is
    already bound to a different environment: a built-in or an earlier
    registration always wins over an import path."""
    bound = registry().get(alias)
    if bound is not None and bound != entry:
        kind = "built-in" if alias in _builtin() else "already registered"
        raise ValueError(f"{source!r} defines env {alias!r}, but that name is {kind} for a "
                         f"different environment; give your environment another `name`")
    _registered[alias] = entry


def get(name: str, allow_import: bool = False) -> EnvEntry:
    """An environment by name: built in, registered with `register`, or
    published by an installed package under the "parley_sim.envs" entry-point
    group. With `allow_import`, also an import path ("my_pkg.envs:ENTRY").

    Only pass `allow_import=True` for a name the user supplied (a config file,
    a command-line flag). Names read from traces or manifests must not be able
    to import and run arbitrary code."""
    envs = registry()
    if name in envs:
        return envs[name]
    if is_import_path(name):
        if not allow_import:
            raise ValueError(f"env {name!r} is an import path; it is only loaded when you name "
                             f"it yourself (e.g. --env {name})")
        entry = _load(name)
        own = entry.env().name  # traces record the env's own name
        _bind_alias(own, entry, name)
        _bind_alias(name, entry, name)
        return entry
    from importlib.metadata import entry_points
    for ep in entry_points(group="parley_sim.envs"):
        if ep.name == name:
            entry = _load(ep.value)
            register(name, entry)
            return entry
    raise ValueError(f"unknown env {name!r} (known: {sorted(envs)}; or give an import "
                     f"path like 'my_pkg.envs:ENTRY')")
