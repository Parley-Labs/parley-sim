"""Paired report across arms, from one or more matched-run directories.

Arms are compared within (seed, focal model): the same scenario and focal
seat, only the counterpart differs. Runs in different directories are joined
on (seed, focal, arm, replicate) after checking that every seed's scenario is
identical. Each comparison's delta sits next to its noise floor: the same
delta between two identically configured replicates of a model arm. A shift
is only worth reading when it clears both its bootstrap interval and that
floor.
"""
from __future__ import annotations

import random
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

from parley_sim import envs
from parley_sim.core.hashing import digest
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


def _mean(xs: list[float]) -> float | None:
    return statistics.fmean(xs) if xs else None


def _arm_order(runs: list[dict[str, Any]], present: set[str]) -> list[str]:
    order = ["soc"]
    for run in runs:
        for arm in run["config"].get("extra_arms") or {}:
            if arm not in order:
                order.append(arm)
    order += sorted(present - set(order) - {"aoc"}) + ["aoc"]
    return [a for a in order if a in present]


def build_report(*out_dirs: str | Path) -> dict[str, Any]:
    runs, env_name = [], None
    cells: dict[tuple[int, str, str, int], dict[str, Any]] = {}
    scenario: dict[int, str] = {}
    decisions: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    status_counts: dict[str, int] = defaultdict(int)
    spend = {"input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0}
    for out_dir in out_dirs:
        out = Path(out_dir)
        run, rows = load_manifest(out)
        runs.append(run)
        if env_name not in (None, run["config"]["env"]):
            raise ValueError("runs use different environments")
        env_name = run["config"]["env"]
        entry = envs.get(env_name)
        for c in rows:
            status_counts[c["status"]] += 1
            for k in spend:
                spend[k] += (c.get("usage") or {}).get(k, 0)
            if c["status"] != "done":
                continue
            lines = read_trace(out / c["trace"])
            header = lines[0]
            fingerprint = digest([header["instance"], header["spec"]["config"],
                                  header["spec"]["focal_seat"]])
            if scenario.setdefault(c["seed"], fingerprint) != fingerprint:
                raise ValueError(f"seed {c['seed']} has a different scenario in {out}")
            key = (c["seed"], c["focal"], c["arm"], c["replicate"])
            if key in cells:
                raise ValueError(f"{key} appears in more than one run")
            cells[key] = {**c, "metrics": entry.episode_metrics(lines)}
            if c["replicate"] == 0:
                decisions[(c["focal"], c["arm"])].extend(entry.decision_rows(lines))
    entry = envs.get(env_name)
    arms = _arm_order(runs, {k[2] for k in cells})
    focals = list(dict.fromkeys(f for run in runs for f in run["config"]["focal_models"]))
    seeds = sorted({k[0] for k in cells})

    def value(seed: int, focal: str, arm: str, r: int, name: str) -> float | None:
        c = cells.get((seed, focal, arm, r))
        return None if c is None else c["metrics"][name]

    means: dict[str, dict[str, dict[str, float | None]]] = {}
    floors: dict[str, dict[str, dict[str, float | None]]] = {}
    comparisons: dict[str, dict[str, dict[str, Any]]] = {}
    for focal in focals:
        means[focal] = {a: {} for a in arms}
        floors[focal] = {a: {} for a in arms}
        for arm in arms:
            for name in entry.metric_kinds:
                vals = [v for s in seeds if (v := value(s, focal, arm, 0, name)) is not None]
                nulls = [abs(b - a) for s in seeds
                         if (a := value(s, focal, arm, 0, name)) is not None
                         and (b := value(s, focal, arm, 1, name)) is not None]
                means[focal][arm][name] = _mean(vals)
                floors[focal][arm][name] = _mean(nulls)
        comparisons[focal] = {}
        for i, a in enumerate(arms):
            for b in arms[i + 1:]:
                table = comparisons[focal][f"{b} vs {a}"] = {}
                for name, kind in entry.metric_kinds.items():
                    deltas = [vb - va for s in seeds
                              if (va := value(s, focal, a, 0, name)) is not None
                              and (vb := value(s, focal, b, 0, name)) is not None]
                    lo, hi = bootstrap_ci(deltas)
                    delta = _mean(deltas)
                    fl = [f for f in (floors[focal][a][name], floors[focal][b][name])
                          if f is not None]
                    floor = max(fl) if fl else None
                    table[name] = {
                        "kind": kind, "n_pairs": len(deltas), "delta_mean": delta,
                        "delta_ci": [lo, hi], "noise_floor": floor,
                        "clears": bool(deltas) and not (lo <= 0 <= hi)
                        and (floor is None or abs(delta) > floor),
                    }

    acceptance: dict[str, dict[str, dict[str, Any]]] = {}
    for (focal, arm), rows in decisions.items():
        table = acceptance.setdefault(focal, {}).setdefault(arm, {})
        for row in rows:
            b = table.setdefault(row["bin"], {"n": 0, "accepted": 0})
            b["n"] += 1
            b["accepted"] += row["accepted"]
    return {"run_ids": [r["run_id"] for r in runs], "env": env_name,
            "cells": dict(status_counts), "spend": spend, "arms": arms,
            "kinds": dict(entry.metric_kinds), "means": means, "noise_floors": floors,
            "comparisons": comparisons, "acceptance_by_offer_quality": acceptance}


def _fmt(v: Any) -> str:
    if v is None or (isinstance(v, float) and v != v):
        return "-"
    return f"{v:+.3f}" if isinstance(v, float) else str(v)


def render(report: dict[str, Any]) -> str:
    arms = report["arms"]
    out = [f"# {' + '.join(report['run_ids'])} ({report['env']})", "",
           "Cells: " + ", ".join(f"{k}={v}" for k, v in sorted(report["cells"].items())),
           "Spend: {input_tokens:,} input + {output_tokens:,} output tokens, "
           "${cost_usd:.2f} reported by provider".format(**report["spend"]), ""]
    for focal, by_arm in report["means"].items():
        out += [f"## {focal}", "", "Means (replicate 0):", "",
                "| metric | kind | " + " | ".join(arms) + " |",
                "|---|---|" + "---|" * len(arms)]
        for name, kind in report["kinds"].items():
            out.append(f"| {name} | {kind} | "
                       + " | ".join(_fmt(by_arm[a][name]) for a in arms) + " |")
        for title, table in report["comparisons"][focal].items():
            out += ["", f"{title}:", "",
                    "| metric | delta [95% CI] | noise floor | n | clears |",
                    "|---|---|---|---|---|"]
            for name, m in table.items():
                lo, hi = m["delta_ci"]
                out.append(f"| {name} | {_fmt(m['delta_mean'])} [{_fmt(lo)}, {_fmt(hi)}] "
                           f"| {_fmt(m['noise_floor'])} | {m['n_pairs']} | "
                           f"{'yes' if m['clears'] else 'no'} |")
        table = report["acceptance_by_offer_quality"].get(focal, {})
        if table:
            out += ["", "Acceptance by offer quality (every faced offer, declines included):", "",
                    "| arm | bin | offers | accepted | rate |", "|---|---|---|---|---|"]
            for arm in arms:
                for b, v in sorted(table.get(arm, {}).items()):
                    out.append(f"| {arm} | {b} | {v['n']} | {v['accepted']} "
                               f"| {v['accepted'] / v['n']:.2f} |")
        out.append("")
    return "\n".join(out)


CLAIM_LABELS = ("understated", "overstated", "accurate", "none")


def cohen_kappa(a: list[str], b: list[str]) -> float:
    n = len(a)
    if n == 0:
        return float("nan")
    cats = sorted(set(a) | set(b))
    observed = sum(x == y for x, y in zip(a, b)) / n
    expected = sum((a.count(c) / n) * (b.count(c) / n) for c in cats)
    return 1.0 if expected == 1 else (observed - expected) / (1 - expected)


def export_claims(out_dir: str | Path, prefix: str | Path, sample: int = 200,
                  seed: int = 0) -> tuple[Path, Path, int]:
    """Write `<prefix>.blind.csv` (sentences and true limits, empty
    `human_label`, no rule output) and `<prefix>.key.csv` (the rules' labels).
    Up to half the sample is sentences the rules flagged, the rest unflagged
    ones, so a rare label still gets enough rows for a stable kappa."""
    import csv
    out = Path(out_dir)
    run, cells = load_manifest(out)
    entry = envs.get(run["config"]["env"])
    if entry.claim_rows is None:
        raise ValueError(f"env {run['config']['env']!r} has no claim classifier")
    rows = [r for c in cells if c["status"] == "done"
            for r in entry.claim_rows(read_trace(out / c["trace"]))]
    rng = random.Random(seed)
    flagged = [r for r in rows if r["auto_label"] != "none"]
    plain = [r for r in rows if r["auto_label"] == "none"]
    rng.shuffle(flagged)
    rng.shuffle(plain)
    take = flagged[: sample // 2]
    take += plain[: sample - len(take)]
    rng.shuffle(take)
    prefix = Path(prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    blind, key = prefix.with_suffix(".blind.csv"), prefix.with_suffix(".key.csv")
    with blind.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "limit_discount_pct", "limit_credit_usd", "sentence", "human_label"])
        for r in take:
            w.writerow([r["id"], r["limit_discount_pct"], r["limit_credit_usd"], r["sentence"], ""])
    with key.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "auto_label", "arm", "label_model"])
        for r in take:
            w.writerow([r["id"], r["auto_label"], r["arm"], r["label_model"]])
    return blind, key, len(take)


def claims_kappa(blind_csv: str | Path, key_csv: str | Path) -> dict[str, Any]:
    import csv
    with Path(key_csv).open() as f:
        auto = {r["id"]: r["auto_label"] for r in csv.DictReader(f)}
    pairs, invalid = [], []
    with Path(blind_csv).open() as f:
        for r in csv.DictReader(f):
            human = (r.get("human_label") or "").strip().lower()
            if not human:
                continue
            if human not in CLAIM_LABELS or r["id"] not in auto:
                invalid.append(r["id"])
                continue
            pairs.append((auto[r["id"]], human))
    a, h = [p[0] for p in pairs], [p[1] for p in pairs]
    confusion = {x: {y: sum(1 for p in pairs if p == (x, y)) for y in CLAIM_LABELS}
                 for x in CLAIM_LABELS}
    return {"n": len(pairs), "invalid": invalid,
            "kappa_understated": cohen_kappa(["understated" if x == "understated" else "other"
                                              for x in a],
                                             ["understated" if x == "understated" else "other"
                                              for x in h]),
            "kappa_all": cohen_kappa(a, h), "confusion_auto_by_human": confusion}
