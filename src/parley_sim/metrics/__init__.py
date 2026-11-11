"""Paired report across arms, from one or more matched-run directories.

Arms are compared within (seed, focal model): the same scenario and focal
seat, only the counterpart differs. Runs in different directories are joined
on (seed, focal, arm, replicate) after checking that every seed's scenario is
identical. Each comparison's delta sits next to its noise floor: the same
delta between two identically configured replicates of a model arm. A shift
is only worth reading when it clears both its bootstrap interval and that
floor.

With equivalence margins, a comparison also gets a verdict: "equivalent" when
the 90% interval of the delta lies inside +/- margin (two one-sided tests at
5%), "different" when it clears its interval and floor, else "inconclusive".
Diversity compares arms on spread rather than means: the across-seed SD of
outcome metrics and the distinct-bigram ratio of the counterpart's opening
messages, each with a paired bootstrap interval over seeds.
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


DISPERSION_METRICS = ("focal_surplus_share", "turns", "cp_first_offer_share")


def bootstrap_ci(values: list[float], n: int = 2000, seed: int = 0,
                 level: float = 0.95) -> tuple[float, float]:
    if len(values) < 2:
        return (float("nan"), float("nan"))
    rng = random.Random(seed)
    means = sorted(statistics.fmean(rng.choices(values, k=len(values))) for _ in range(n))
    tail = (1 - level) / 2
    return means[int(tail * n)], means[int((1 - tail) * n) - 1]


def _paired_stat_ci(pairs: list[tuple[Any, Any]], stat, n: int = 2000,
                    seed: int = 0) -> tuple[float | None, float, float]:
    """Delta stat(b) - stat(a) over seed-paired samples, with a 95% bootstrap
    interval that resamples seeds (keeping each seed's pair together)."""
    if len(pairs) < 3:
        return None, float("nan"), float("nan")
    point = stat([b for _, b in pairs]) - stat([a for a, _ in pairs])
    rng = random.Random(seed)
    boots = []
    for _ in range(n):
        sample = rng.choices(pairs, k=len(pairs))
        boots.append(stat([b for _, b in sample]) - stat([a for a, _ in sample]))
    boots.sort()
    return point, boots[int(0.025 * n)], boots[int(0.975 * n) - 1]


def _paired_permutation_p(pairs: list[tuple[Any, Any]], stat, n: int = 2000,
                          seed: int = 0) -> tuple[float | None, float]:
    """Delta stat(b) - stat(a) and a two-sided p-value from randomly swapping
    the arms within each seed's pair. Used for statistics a bootstrap would
    bias (resampling duplicates texts, which deflates distinct-bigram ratios)."""
    if len(pairs) < 3:
        return None, float("nan")
    point = stat([b for _, b in pairs]) - stat([a for a, _ in pairs])
    rng = random.Random(seed)
    hits = 0
    for _ in range(n):
        swapped = [(b, a) if rng.random() < 0.5 else (a, b) for a, b in pairs]
        d = stat([b for _, b in swapped]) - stat([a for a, _ in swapped])
        hits += abs(d) >= abs(point) - 1e-12
    return point, (hits + 1) / (n + 1)


def distinct2(texts: list[str]) -> float:
    """Unique word bigrams / all word bigrams across a set of messages."""
    grams = [g for t in texts for g in zip(t.lower().split(), t.lower().split()[1:])]
    return len(set(grams)) / len(grams) if grams else float("nan")


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


def build_report(*out_dirs: str | Path,
                 margins: dict[str, float] | None = None) -> dict[str, Any]:
    margins = margins or {}
    runs, env_name = [], None
    cells: dict[tuple[int, str, str, int], dict[str, Any]] = {}
    scenario: dict[int, str] = {}
    decisions: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    openings: dict[tuple[str, str], dict[int, str]] = defaultdict(dict)
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
                if entry.counterpart_opening is not None:
                    text = entry.counterpart_opening(lines)
                    if text:
                        openings[(c["focal"], c["arm"])][c["seed"]] = text
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
                    lo90, hi90 = bootstrap_ci(deltas, level=0.90)
                    delta = _mean(deltas)
                    fl = [f for f in (floors[focal][a][name], floors[focal][b][name])
                          if f is not None]
                    floor = max(fl) if fl else None
                    clears = bool(deltas) and not (lo <= 0 <= hi) and (
                        floor is None or abs(delta) > floor)
                    margin = margins.get(name)
                    if margin is not None and bool(deltas) and -margin <= lo90 and hi90 <= margin:
                        verdict = "equivalent"
                    elif clears:
                        verdict = "different"
                    else:
                        verdict = "inconclusive" if margin is not None else "-"
                    table[name] = {
                        "kind": kind, "n_pairs": len(deltas), "delta_mean": delta,
                        "delta_ci": [lo, hi], "delta_ci90": [lo90, hi90],
                        "noise_floor": floor, "margin": margin, "clears": clears,
                        "verdict": verdict,
                    }

    diversity: dict[str, dict[str, Any]] = {}
    for focal in focals:
        per_arm: dict[str, dict[str, float | None]] = {}
        for arm in arms:
            row: dict[str, float | None] = {}
            for name in DISPERSION_METRICS:
                if name in entry.metric_kinds:
                    vals = [v for s in seeds if (v := value(s, focal, arm, 0, name)) is not None]
                    row[f"sd_{name}"] = statistics.stdev(vals) if len(vals) > 1 else None
            texts = list(openings[(focal, arm)].values())
            row["opening_distinct2"] = distinct2(texts) if texts else None
            per_arm[arm] = row
        comps: dict[str, dict[str, Any]] = {}
        for i, a in enumerate(arms):
            for b in arms[i + 1:]:
                out_c: dict[str, Any] = {}
                for name in DISPERSION_METRICS:
                    if name not in entry.metric_kinds:
                        continue
                    pairs = [(va, vb) for s in seeds
                             if (va := value(s, focal, a, 0, name)) is not None
                             and (vb := value(s, focal, b, 0, name)) is not None]
                    d, lo, hi = _paired_stat_ci(pairs, statistics.stdev)
                    out_c[f"sd_{name}"] = {"delta": d, "ci": [lo, hi], "n": len(pairs)}
                pairs = [(openings[(focal, a)][s], openings[(focal, b)][s]) for s in seeds
                         if s in openings[(focal, a)] and s in openings[(focal, b)]]
                d, pval = _paired_permutation_p(pairs, distinct2)
                out_c["opening_distinct2"] = {"delta": d, "p": pval, "n": len(pairs)}
                comps[f"{b} vs {a}"] = out_c
        diversity[focal] = {"per_arm": per_arm, "comparisons": comps}

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
            "comparisons": comparisons, "diversity": diversity, "margins": margins,
            "acceptance_by_offer_quality": acceptance}


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
                    "| metric | delta [95% CI] | 90% CI | margin | noise floor | n | verdict |",
                    "|---|---|---|---|---|---|---|"]
            for name, m in table.items():
                lo, hi = m["delta_ci"]
                lo90, hi90 = m["delta_ci90"]
                out.append(f"| {name} | {_fmt(m['delta_mean'])} [{_fmt(lo)}, {_fmt(hi)}] "
                           f"| [{_fmt(lo90)}, {_fmt(hi90)}] | {_fmt(m['margin'])} "
                           f"| {_fmt(m['noise_floor'])} | {m['n_pairs']} | {m['verdict']} |")
        div = report["diversity"].get(focal)
        if div:
            cols = list(next(iter(div["per_arm"].values())))
            out += ["", "Diversity (spread across seeds; opening-message variety):", "",
                    "| arm | " + " | ".join(cols) + " |", "|---|" + "---|" * len(cols)]
            for arm, row in div["per_arm"].items():
                out.append(f"| {arm} | " + " | ".join(_fmt(row[c]) for c in cols) + " |")
            for title, comp in div["comparisons"].items():
                out.append(f"\n{title}: " + "; ".join(
                    f"{k} {_fmt(v['delta'])} " + (f"(p={v['p']:.3f})" if "p" in v else
                                                  f"[{_fmt(v['ci'][0])}, {_fmt(v['ci'][1])}]")
                    for k, v in comp.items()))
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
                  seed: int = 0, exclude: set[str] | None = None) -> tuple[Path, Path, int]:
    """Write `<prefix>.blind.csv` (sentences and true limits, empty
    `human_label`, no rule output) and `<prefix>.key.csv` (the rules' labels,
    plus each label's total count in the run so agreement can be reweighted).

    Stratified by the rules' label: half the sample is sentences the rules
    flagged, split evenly across flagged labels with any rare label taken
    whole (a bluff may occur a handful of times in thousands of sentences);
    the other half is unflagged sentences, where the rules' misses would be."""
    import csv
    out = Path(out_dir)
    run, cells = load_manifest(out)
    entry = envs.get(run["config"]["env"])
    if entry.claim_rows is None:
        raise ValueError(f"env {run['config']['env']!r} has no claim classifier")
    rows = [r for c in cells if c["status"] == "done"
            for r in entry.claim_rows(read_trace(out / c["trace"]))
            if r["id"] not in (exclude or set())]
    rng = random.Random(seed)
    by_label: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        by_label[r["auto_label"]].append(r)
    for group in by_label.values():
        rng.shuffle(group)
    totals = {label: len(group) for label, group in by_label.items()}
    flagged = sorted((l for l in by_label if l != "none"), key=lambda l: totals[l])
    take: list[dict[str, Any]] = []
    budget = sample // 2
    for i, label in enumerate(flagged):
        quota = (budget - len(take)) // (len(flagged) - i)
        take += by_label[label][:quota]
    take += by_label.get("none", [])[: sample - len(take)]
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
        w.writerow(["id", "auto_label", "arm", "label_model", "label_total_in_run"])
        for r in take:
            w.writerow([r["id"], r["auto_label"], r["arm"], r["label_model"],
                        totals[r["auto_label"]]])
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
