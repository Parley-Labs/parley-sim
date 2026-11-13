"""Canonical form of a model spec for identity checks.

Run manifests store each lineup entry as it was written at the time, so an
older manifest lacks fields added later. Fields listed here are dropped while
at their default, so an old manifest, a resumed run and a new run that mean
the same model compare equal, and a non-default value still differs.
"""
from __future__ import annotations

from typing import Any

# Fields added to ModelSpec after runs were published, with their defaults.
ADDED_DEFAULTS: dict[str, Any] = {"conversation": False}


def canonical_model_spec(spec: dict[str, Any] | None) -> dict[str, Any] | None:
    if spec is None:
        return None
    return {k: v for k, v in spec.items()
            if not (k in ADDED_DEFAULTS and v == ADDED_DEFAULTS[k])}
