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
import re
from collections.abc import Callable
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from parley_sim import envs
from parley_sim.core.protocol import Seat
from parley_sim.drivers.local import run_episode
from parley_sim.drivers.models import (
    Breaker,
    ModelSeat,
    ModelSpec,
    ProviderClient,
    ScriptedSeat,
    build_client,
)
from parley_sim.runtime.episode import EpisodeSpec
from parley_sim.trace import JsonlWriter, is_complete, iter_trace

ClientFactory = Callable[[ModelSpec], ProviderClient]
_ID = r"^[A-Za-z0-9._-]+$"
_OPERATIONAL = ("concurrency", "max_total_tokens", "max_refusals")


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
    # More model-counterpart arms, run between "soc" and "aoc" (e.g. a UserSim
    # persona arm); each gets 1 + self_null_replicates cells like "aoc".
    extra_arms: dict[str, list[str]] = {}
    # Arms to execute; others are left to another run directory on the same
    # seeds, which `parley-sim report` joins on (seed, focal, arm, replicate).
    run_arms: list[str] | None = None
    # usersim panel parquet (or JSONL of persona dicts) for lineup entries
    # with `persona: usersim`; one persona per seed, shared across arms.
    persona_file: str | None = None
    self_null_replicates: int = Field(default=1, ge=0)
    concurrency: int = Field(default=4, ge=1)
    max_total_tokens: int | None = Field(default=None, ge=1)
    # Stop the whole run after this many provider refusals (or any 401/403).
    max_refusals: int = Field(default=3, ge=1)

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
    def _references(self) -> MatchedConfig:
        entry = envs.get(self.env)
        names = [m.name for m in self.lineup]
        if len(set(names)) != len(names):
            raise ValueError("lineup names must be unique")
        model_refs = self.focal_models + self.adaptive_counterparts + [
            n for arm in self.extra_arms.values() for n in arm]
        missing = [n for n in model_refs if n not in names]
        if missing:
            raise ValueError(f"not in lineup: {missing}")
        for arm, members in self.extra_arms.items():
            if arm in ("soc", "aoc") or not re.fullmatch(r"[a-z][a-z0-9_]*", arm) or not members:
                raise ValueError(f"extra arm {arm!r}: needs a new lowercase name and members")
        unknown_arms = set(self.run_arms or []) - set(self.arm_names())
        if unknown_arms:
            raise ValueError(f"run_arms not in {self.arm_names()}: {sorted(unknown_arms)}")
        if any(m.persona for m in self.lineup) and not self.persona_file:
            raise ValueError("lineup entries with a persona need persona_file")
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

    def arm_names(self) -> list[str]:
        return ["soc", *self.extra_arms, "aoc"]


class Cell(BaseModel):
    model_config = ConfigDict(extra="forbid")

    cell_key: str
    pair_key: str
    arm: str
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
            seats = {s: f"{prefix}:{names[(i + j) % len(names)]}" for j, s in enumerate(others)}  # noqa: B023
            return {focal_seat: f"model:{focal}", **seats}

        model_arms = {**cfg.extra_arms, "aoc": cfg.adaptive_counterparts}
        for focal in cfg.focal_models:
            pair_key = f"{cfg.run_id}-{seed}-{focal}"
            cells.append(Cell(cell_key=f"{pair_key}-soc-r0", pair_key=pair_key, arm="soc",
                              seed=seed, focal=focal, focal_seat=focal_seat, replicate=0,
                              seats=fill(cfg.scripted_counterparts, "policy", focal)))
            for arm, members in model_arms.items():
                for r in range(1 + cfg.self_null_replicates):
                    cells.append(Cell(cell_key=f"{pair_key}-{arm}-r{r}", pair_key=pair_key,
                                      arm=arm, seed=seed, focal=focal, focal_seat=focal_seat,
                                      replicate=r, seats=fill(members, "model", focal)))
    if cfg.run_arms is not None:
        cells = [c for c in cells if c.arm in cfg.run_arms]
    return cells


def _usage(trace: Path) -> dict[str, float]:
    total: dict[str, float] = {"input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0}
    for line in iter_trace(trace):
        usage = ((line.get("raw") or {}).get("usage") or {}) if line["type"] == "decision" else {}
        for k in total:
            total[k] += usage.get(k, 0)
    return total


def _tokens(usage: dict[str, float]) -> int:
    return int(usage.get("input_tokens", 0) + usage.get("output_tokens", 0))


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
    breaker = Breaker(cfg.max_refusals)
    spent = {"tokens": sum(_tokens(r["usage"]) for r in done.values())}
    lock = asyncio.Lock()
    sem = asyncio.Semaphore(cfg.concurrency)
    fh = manifest.open("a")

    def append(row: dict[str, Any]) -> None:
        fh.write(json.dumps(row, sort_keys=True) + "\n")
        fh.flush()

    personas = None
    if cfg.persona_file and any(m.persona for m in cfg.lineup):
        from parley_sim.drivers.usersim_persona import load_personas
        personas = load_personas(cfg.persona_file)

    def text_for(spec: ModelSpec, seed: int):
        if spec.persona == "usersim":
            from parley_sim.drivers.usersim_persona import UserSimPersonaText, persona_for_seed
            return UserSimPersonaText(entry.text(), persona_for_seed(personas, seed))
        return entry.text()

    def agent_for(seat: Seat, name: str, seed: int) -> ModelSeat | ScriptedSeat:
        spec = lineup[name]
        if spec.provider == "scripted":
            return ScriptedSeat(spec, text_for(spec, seed), entry.make_policy(spec.model, seat))
        return ModelSeat(spec, text_for(spec, seed), client_factory(spec), breaker)

    async def run_cell(cell: Cell) -> None:
        if cell.cell_key in done:
            return
        async with sem:
            if breaker.tripped is not None:
                async with lock:
                    append({"type": "cell", **cell.model_dump(), "status": "skipped_halted",
                            "trace": None, "usage": {"input_tokens": 0, "output_tokens": 0},
                            "error": f"run halted: {breaker.tripped}"})
                return
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
                agents = {seat: agent_for(seat, label.split(":", 1)[1], cell.seed)
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
                spent["tokens"] += _tokens(usage)
                append(row)

    try:
        await asyncio.gather(*(run_cell(c) for c in cells))
    finally:
        fh.close()
    return manifest
