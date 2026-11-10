"""Matched scripted-vs-adaptive runs.

For every (seed, focal model) the grid holds one SOC cell (scripted
counterparts) and 1 + `self_null_replicates` AOC cells (adaptive model
counterparts), all with the same seed, scenario and focal seat. The only thing
that differs inside a `pair_key` is what fills the non-focal seats, so the
SOC-AOC delta is attributable to counterpart adaptivity; AOC replicate r vs r+1
is the same configuration twice, so its delta is the noise floor.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Callable, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from parley_sim import envs
from parley_sim.core.protocol import Seat
from parley_sim.drivers.local import run_episode
from parley_sim.drivers.models import ModelSeat, ModelSpec, ProviderClient, ScriptedSeat, build_client
from parley_sim.runtime.episode import EpisodeSpec
from parley_sim.trace import JsonlWriter, is_complete, iter_trace

ClientFactory = Callable[[ModelSpec], ProviderClient]
_ID = r"^[A-Za-z0-9._-]+$"
_OPERATIONAL = ("concurrency", "max_total_tokens")


def parse_seeds(spec: str) -> range:
    start, sep, stop = spec.partition(":")
    if not sep:
        raise ValueError(f"seeds must be 'start:stop', got {spec!r}")
    return range(int(start), int(stop))


class MatchedConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    run_id: str = Field(pattern=_ID)
    env: str = "support_desk"
    env_config: dict[str, Any] = {}
    seeds: str
    focal_seat: str | None = None
    lineup: list[ModelSpec]
    focal_models: list[str]
    scripted_counterparts: list[str]
    adaptive_counterparts: list[str]
    self_null_replicates: int = Field(default=1, ge=0)
    concurrency: int = Field(default=4, ge=1)
    max_total_tokens: int | None = Field(default=None, ge=1)

    @field_validator("seeds")
    @classmethod
    def _seeds(cls, v: str) -> str:
        if len(parse_seeds(v)) == 0:
            raise ValueError("seed range is empty")
        return v

    @field_validator("focal_models", "scripted_counterparts", "adaptive_counterparts")
    @classmethod
    def _non_empty_unique(cls, v: list[str]) -> list[str]:
        if not v or len(set(v)) != len(v):
            raise ValueError("must be non-empty and without duplicates")
        return v

    @model_validator(mode="after")
    def _references(self) -> "MatchedConfig":
        entry = envs.get(self.env)
        names = [m.name for m in self.lineup]
        if len(set(names)) != len(names):
            raise ValueError("lineup names must be unique")
        missing = [n for n in self.focal_models + self.adaptive_counterparts if n not in names]
        if missing:
            raise ValueError(f"not in lineup: {missing}")
        instance = entry.env().generate(parse_seeds(self.seeds)[0], self.env_config)
        seats = entry.env().seats(instance)
        focal = self.focal_seat or entry.focal_seat
        if focal not in seats:
            raise ValueError(f"focal_seat {focal!r} is not one of {list(seats)}")
        for seat in seats:
            if seat != focal:
                for name in self.scripted_counterparts:
                    entry.make_policy(name, seat)  # raises on unknown / wrong seat
        return self


class Cell(BaseModel):
    model_config = ConfigDict(extra="forbid")

    cell_key: str
    pair_key: str
    arm: Literal["soc", "aoc"]
    seed: int
    focal: str
    focal_seat: Seat
    replicate: int
    seats: dict[Seat, str]


def load_config(path: str | Path) -> MatchedConfig:
    return MatchedConfig(**yaml.safe_load(Path(path).read_text()))


def config_hash(cfg: MatchedConfig) -> str:
    data = cfg.model_dump()
    for k in _OPERATIONAL:
        data.pop(k)
    return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()


def build_grid(cfg: MatchedConfig) -> list[Cell]:
    """Counterparts rotate across seeds (seed index i, non-focal seat j gets
    names[(i + j) % len(names)]), so every scripted persona and every adaptive
    counterpart is covered even with one non-focal seat."""
    entry = envs.get(cfg.env)
    env = entry.env()
    focal_seat = cfg.focal_seat or entry.focal_seat
    cells: list[Cell] = []
    for i, seed in enumerate(parse_seeds(cfg.seeds)):
        others = [s for s in env.seats(env.generate(seed, cfg.env_config)) if s != focal_seat]

        def fill(names: list[str], prefix: str, focal: str) -> dict[Seat, str]:
            seats = {s: f"{prefix}:{names[(i + j) % len(names)]}" for j, s in enumerate(others)}
            return {focal_seat: f"model:{focal}", **seats}

        for focal in cfg.focal_models:
            pair_key = f"{cfg.run_id}-{seed}-{focal}"
            cells.append(Cell(cell_key=f"{pair_key}-soc-r0", pair_key=pair_key, arm="soc",
                              seed=seed, focal=focal, focal_seat=focal_seat, replicate=0,
                              seats=fill(cfg.scripted_counterparts, "policy", focal)))
            for r in range(1 + cfg.self_null_replicates):
                cells.append(Cell(cell_key=f"{pair_key}-aoc-r{r}", pair_key=pair_key, arm="aoc",
                                  seed=seed, focal=focal, focal_seat=focal_seat, replicate=r,
                                  seats=fill(cfg.adaptive_counterparts, "model", focal)))
    return cells


def _usage(trace: Path) -> dict[str, int]:
    total = {"input_tokens": 0, "output_tokens": 0}
    for line in iter_trace(trace):
        usage = ((line.get("raw") or {}).get("usage") or {}) if line["type"] == "decision" else {}
        for k in total:
            total[k] += usage.get(k, 0)
    return total


def _write_atomic(path: Path, rows: list[dict[str, Any]]) -> None:
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


async def run_matched(cfg: MatchedConfig, out_dir: str | Path,
                      client_factory: ClientFactory = build_client) -> Path:
    """Run every cell not already complete. Resumable: a cell is skipped only
    if its manifest row says done and its trace has a footer. Refuses to resume
    into a directory built from a different experiment definition."""
    out = Path(out_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    manifest = out / "manifest.jsonl"
    chash = config_hash(cfg)
    cells = build_grid(cfg)
    run_row = {"type": "run", "run_id": cfg.run_id, "config_hash": chash,
               "config": cfg.model_dump()}

    done: dict[str, dict[str, Any]] = {}
    if manifest.exists():
        rows = list(iter_trace(manifest))
        prior = next((r for r in rows if r["type"] == "run"), None)
        if prior is None or prior["config_hash"] != chash:
            raise RuntimeError(f"{manifest} was built from a different config; use a new --out")
        for r in rows:
            if r["type"] == "cell" and r["status"] == "done" and is_complete(out / r["trace"]):
                done[r["cell_key"]] = r
    _write_atomic(manifest, [run_row, *done.values()])

    entry = envs.get(cfg.env)
    lineup = {m.name: m for m in cfg.lineup}
    spent = {"tokens": sum(sum(r["usage"].values()) for r in done.values())}
    lock = asyncio.Lock()
    sem = asyncio.Semaphore(cfg.concurrency)
    fh = manifest.open("a")

    def append(row: dict[str, Any]) -> None:
        fh.write(json.dumps(row, sort_keys=True) + "\n")
        fh.flush()

    def agent_for(seat: Seat, name: str) -> ModelSeat | ScriptedSeat:
        spec = lineup[name]
        if spec.provider == "scripted":
            return ScriptedSeat(spec, entry.text(), entry.make_policy(spec.model, seat))
        return ModelSeat(spec, entry.text(), client_factory(spec))

    async def run_cell(cell: Cell) -> None:
        if cell.cell_key in done:
            return
        async with sem:
            async with lock:
                if cfg.max_total_tokens is not None and spent["tokens"] >= cfg.max_total_tokens:
                    append({"type": "cell", **cell.model_dump(), "status": "skipped_budget",
                            "trace": None, "usage": {"input_tokens": 0, "output_tokens": 0}})
                    return
            trace = out / "traces" / f"{cell.cell_key}.jsonl"
            status, error = "done", None
            try:
                spec = EpisodeSpec(seed=cell.seed, seats=cell.seats, config=cfg.env_config,
                                   focal_seat=cell.focal_seat, arm=cell.arm,
                                   pair_key=cell.pair_key, cell_key=cell.cell_key,
                                   replicate=cell.replicate)
                agents = {seat: agent_for(seat, label.split(":", 1)[1])
                          for seat, label in cell.seats.items() if label.startswith("model:")}
                with JsonlWriter(trace) as writer:
                    await run_episode(entry, spec, agents, sink=writer)
            except Exception as exc:  # one bad cell never takes down the grid
                status, error = "errored", repr(exc)
            usage = _usage(trace) if trace.exists() else {"input_tokens": 0, "output_tokens": 0}
            row = {"type": "cell", **cell.model_dump(), "status": status,
                   "trace": str(trace.relative_to(out)), "usage": usage}
            if error:
                row["error"] = error
            async with lock:
                spent["tokens"] += sum(usage.values())
                append(row)

    try:
        await asyncio.gather(*(run_cell(c) for c in cells))
    finally:
        fh.close()
    return manifest
