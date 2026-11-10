"""Paired SOC/AOC report from a matched-run directory.

Each metric's scripted-vs-adaptive delta is computed within `pair_key` (same
seed, scenario, focal seat) and shown next to the self-null noise floor: the
same delta between two identically configured AOC replicates. A shift is only
worth reading when it clears both its bootstrap interval and that floor.
"""
from __future__ import annotations

import random
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

from parley_sim import envs
from parley_sim.trace import iter_trace, read_trace


def load_manifest(out_dir: str | Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    rows = list(iter_trace(Path(out_dir) / "manifest.jsonl"))
    run = next(r for r in rows if r["type"] == "run")
    latest: dict[str, dict[str, Any]] = {}
    for r in rows:
        if r["type"] == "cell":
            latest[r["cell_key"]] = r
    return run, list(latest.values())


def bootstrap_ci(values: list[float], n: int = 2000, seed: int = 0) -> tuple[float, float]:
    if len(values) < 2:
        return (float("nan"), float("nan"))
    rng = random.Random(seed)
    means = sorted(statistics.fmean(rng.choices(values, k=len(values))) for _ in range(n))
    return means[int(0.025 * n)], means[int(0.975 * n) - 1]


def build_report(out_dir: str | Path) -> dict[str, Any]:
    out = Path(out_dir)
    run, cells = load_manifest(out)
    entry = envs.get(run["config"]["env"])
    by_key: dict[str, dict[str, Any]] = {}
    decisions: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    status_counts: dict[str, int] = defaultdict(int)
    for c in cells:
        status_counts[c["status"]] += 1
        if c["status"] != "done":
            continue
        lines = read_trace(out / c["trace"])
        by_key[c["cell_key"]] = {**c, "metrics": entry.episode_metrics(lines)}
        if c["arm"] == "soc" or c["replicate"] == 0:
            decisions[(c["focal"], c["arm"])].extend(entry.decision_rows(lines))

    def cell(pair: str, arm: str, r: int) -> dict[str, Any] | None:
        return by_key.get(f"{pair}-{arm}-r{r}")

    focals = run["config"]["focal_models"]
    pairs_by_focal: dict[str, set[str]] = defaultdict(set)
    for c in by_key.values():
        pairs_by_focal[c["focal"]].add(c["pair_key"])

    metrics: dict[str, dict[str, Any]] = {}
    for focal in focals:
        metrics[focal] = {}
        for name, kind in entry.metric_kinds.items():
            soc, aoc, deltas, nulls = [], [], [], []
            for pair in sorted(pairs_by_focal[focal]):
                s, a, a1 = cell(pair, "soc", 0), cell(pair, "aoc", 0), cell(pair, "aoc", 1)
                sv = s["metrics"][name] if s else None
                av = a["metrics"][name] if a else None
                if sv is not None:
                    soc.append(sv)
                if av is not None:
                    aoc.append(av)
                if sv is not None and av is not None:
                    deltas.append(av - sv)
                a1v = a1["metrics"][name] if a1 else None
                if av is not None and a1v is not None:
                    nulls.append(abs(a1v - av))
            lo, hi = bootstrap_ci(deltas)
            delta = statistics.fmean(deltas) if deltas else None
            floor = statistics.fmean(nulls) if nulls else None
            metrics[focal][name] = {
                "kind": kind, "n_pairs": len(deltas),
                "soc_mean": statistics.fmean(soc) if soc else None,
                "aoc_mean": statistics.fmean(aoc) if aoc else None,
                "delta_mean": delta, "delta_ci": [lo, hi],
                "noise_floor": floor, "n_null": len(nulls),
                "clears": bool(deltas) and not (lo <= 0 <= hi)
                and (floor is None or abs(delta) > floor),
            }

    acceptance: dict[str, dict[str, dict[str, Any]]] = {}
    for (focal, arm), rows in sorted(decisions.items()):
        table = acceptance.setdefault(focal, {}).setdefault(arm, {})
        for row in rows:
            b = table.setdefault(row["bin"], {"n": 0, "accepted": 0})
            b["n"] += 1
            b["accepted"] += row["accepted"]
    return {"run_id": run["run_id"], "env": run["config"]["env"], "cells": dict(status_counts),
            "metrics": metrics, "acceptance_by_offer_quality": acceptance}


def _fmt(v: Any) -> str:
    if v is None or (isinstance(v, float) and v != v):
        return "-"
    return f"{v:+.3f}" if isinstance(v, float) else str(v)


def render(report: dict[str, Any]) -> str:
    out = [f"# {report['run_id']} ({report['env']})", "",
           "Cells: " + ", ".join(f"{k}={v}" for k, v in sorted(report["cells"].items())), ""]
    for focal, metrics in report["metrics"].items():
        out += [f"## {focal}", "",
                "| metric | kind | SOC | AOC | AOC-SOC [95% CI] | noise floor | n | clears |",
                "|---|---|---|---|---|---|---|---|"]
        for name, m in metrics.items():
            lo, hi = m["delta_ci"]
            out.append(f"| {name} | {m['kind']} | {_fmt(m['soc_mean'])} | {_fmt(m['aoc_mean'])} "
                       f"| {_fmt(m['delta_mean'])} [{_fmt(lo)}, {_fmt(hi)}] "
                       f"| {_fmt(m['noise_floor'])} | {m['n_pairs']} | "
                       f"{'yes' if m['clears'] else 'no'} |")
        table = report["acceptance_by_offer_quality"].get(focal, {})
        if table:
            out += ["", "Acceptance by offer quality (every faced offer, declines included):", "",
                    "| arm | bin | offers | accepted | rate |", "|---|---|---|---|---|"]
            for arm, bins in table.items():
                for b, v in bins.items():
                    out.append(f"| {arm} | {b} | {v['n']} | {v['accepted']} "
                               f"| {v['accepted'] / v['n']:.2f} |")
        out.append("")
    return "\n".join(out)
