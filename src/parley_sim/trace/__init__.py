"""JSONL trace format.

One file per episode. Line types, in order: one `header`, then `decision` and
`event` lines interleaved by `seq`, then one `footer`. A footer is written even
when an episode errors, so a missing footer always means a truncated file.

Every decision is recorded, scripted or external, accepted offer or declined,
together with the decision context (what was being answered) and, for a
fallback, the reason it was taken.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator

from parley_sim.core.hashing import canonical_json

SCHEMA_VERSION = "1"


class JsonlWriter:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._f = self.path.open("w")

    def __call__(self, line: dict[str, Any]) -> None:
        self._f.write(canonical_json(line) + "\n")
        self._f.flush()

    def close(self) -> None:
        self._f.close()

    def __enter__(self) -> "JsonlWriter":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def iter_trace(path: str | Path) -> Iterator[dict[str, Any]]:
    with Path(path).open() as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def read_trace(path: str | Path) -> list[dict[str, Any]]:
    return list(iter_trace(path))


def is_complete(path: str | Path) -> bool:
    try:
        lines = read_trace(path)
    except (OSError, json.JSONDecodeError):
        return False
    return bool(lines) and lines[-1].get("type") == "footer"
