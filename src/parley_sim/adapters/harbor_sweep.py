"""Persona sweep for Harbor's simulated-user trials (experimental).

Harbor can run any task with a simulated user: a user agent gets the task's
instruction and talks to the agent under test, which learns the task only
through that conversation (`harbor run --user-agent ... --user-persona-path P`).
The persona file sets who the user is; the task, its goal and its verifier stay
the same. That makes the persona a clean treatment: this module runs the same
tasks with different personas and asks whether the agent's reward changes.

Arms come from persona file names:
  (no file)          "harbor-default"  Harbor's built-in persona
  usersim-sNN.md     "usersim"         a NeMo UserSim persona from your panel
  minimal.md         "minimal"         a terse user who answers only what is asked
  <arm>.md, <arm>-sNN.md               any persona you write

The matched unit is the task: each arm's reward is averaged over its trials of
a task, and arms are compared on the per-task means, as parley-sim compares
arms on seeds. Persona files built from a UserSim panel hold persona data; keep
them out of version control.
"""

from __future__ import annotations

import json
import math
import re
import shlex
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

from parley_sim.metrics import _paired_permutation_p, bootstrap_ci

DEFAULT_ARM = "harbor-default"
_SEEDED = re.compile(r"^(?P<arm>.+?)-s(?P<seed>\d+)$")

INVARIANTS = """\
You are a person who wants an assistant to get a task done. Stay in character:
never say you are an AI, role-playing, or following instructions, and never
mention tests, grading or benchmarks. Your goal is given after this persona and
is private: the assistant learns it only from what you tell it. If something
isn't covered by your goal, say it's their call rather than inventing details.
"""

MINIMAL_PERSONA = INVARIANTS + """
Be terse. Answer only what you are asked, in a few words. Don't volunteer
requirements, don't explain your reasons, and don't ask questions back.
"""


def write_personas(panel: str | Path, out: str | Path, n: int, locale: str = "en_US") -> list[Path]:
    """`usersim-s00.md` ... from the first `n` personas of a UserSim panel,
    plus `minimal.md`. Needs the `usersim` extra."""
    from parley_sim.drivers.usersim_persona import load_personas, usersim_block

    personas = load_personas(panel)
    if n > len(personas):
        raise ValueError(f"the panel has {len(personas)} personas, fewer than {n}")
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    paths = []
    for i, persona in enumerate(personas[:n]):
        block, meta = usersim_block(persona, locale)
        path = out / f"usersim-s{i:02d}.md"
        path.write_text(f"{INVARIANTS}\n{block}\n")
        (out / f"usersim-s{i:02d}.json").write_text(json.dumps(
            {k: meta[k] for k in ("interaction_style", "disclosure_style", "ocean_labels") if k in meta},
            sort_keys=True))
        paths.append(path)
    minimal = out / "minimal.md"
    minimal.write_text(MINIMAL_PERSONA)
    return [*paths, minimal]


def arm_of(persona_path: str | None) -> tuple[str, int | None]:
    """(arm, seed) from a persona file path; Harbor's default persona if none."""
    if not persona_path:
        return DEFAULT_ARM, None
    stem = Path(persona_path).stem
    m = _SEEDED.match(stem)
    return (m["arm"], int(m["seed"])) if m else (stem, None)


def write_plan(tasks: list[str], personas: list[str | None], *, agent: str, model: str,
               user_agent: str, user_model: str, jobs: str, replicates: int = 1,
               extra: str = "") -> str:
    """A shell script with one `harbor run` per task and persona (None: Harbor's
    default persona), each with `replicates` attempts."""
    lines = ["#!/bin/bash", "set -uo pipefail", ""]
    for task in tasks:
        for persona in personas:
            arm, seed = arm_of(persona)
            name = re.sub(r"[^A-Za-z0-9._-]+", "-", f"{Path(task).name}__{arm}"
                          + (f"-s{seed:02d}" if seed is not None else ""))
            cmd = ["harbor", "run", "-p", task, "-a", agent, "-m", model,
                   "--user-agent", user_agent, "--user-model", user_model, "--bridge", "acp",
                   "-k", str(replicates), "-o", jobs, "--job-name", name]
            if persona:
                cmd += ["--user-persona-path", persona]
            lines.append(" ".join(shlex.quote(c) for c in cmd) + (f" {extra}" if extra else ""))
    return "\n".join(lines) + "\n"


def collect(job_dirs: list[str | Path], reward_key: str = "reward") -> list[dict[str, Any]]:
    """One row per Harbor trial that has a user agent: task, agent, model,
    user agent and model, arm, seed, reward (None if the trial produced none)."""
    rows = []
    for d in job_dirs:
        for result_path in sorted(Path(d).rglob("result.json")):
            trial = result_path.parent
            config_path = trial / "config.json"
            if not config_path.exists():
                continue  # a job-level result.json
            config = json.loads(config_path.read_text())
            user = config.get("user_agent")
            if not user:
                continue
            result = json.loads(result_path.read_text())
            rewards = (result.get("verifier_result") or {}).get("rewards") or {}
            reward = rewards.get(reward_key)
            if reward is None and reward_key == "reward" and len(rewards) == 1:
                reward = next(iter(rewards.values()))  # a task with one, differently named reward
            arm, seed = arm_of(user.get("user_persona_path"))
            agent = config.get("agent") or {}
            task = config.get("task") or {}
            rows.append({"task": result.get("task_name") or task.get("name") or task.get("path"),
                         "agent": agent.get("name"), "model": agent.get("model_name"),
                         "user_agent": user.get("name"), "user_model": user.get("model_name"),
                         "arm": arm, "seed": seed,
                         "reward": None if reward is None else float(reward),
                         "exception": bool(result.get("exception_info")),
                         "trial": str(trial)})
    return rows


def sweep_report(rows: list[dict[str, Any]], baseline: str = DEFAULT_ARM,
                 margin: float | None = None) -> dict[str, Any]:
    """Per agent, model, user agent and user model: each arm's mean reward
    (the mean of its per-task means), and every other arm against `baseline`
    on the tasks both ran. Only the persona differs within a group, so a
    different user agent or user model is never compared as a persona. A trial
    without a reward counts as 0 (Harbor's convention for a failed run) and is
    reported."""
    groups: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        groups[(r["agent"], r["model"], r["user_agent"], r["user_model"])].append(r)
    out: dict[str, Any] = {"baseline": baseline, "margin": margin, "groups": []}
    for (agent, model, user_agent, user_model), rs in sorted(
            groups.items(), key=lambda kv: tuple(str(x) for x in kv[0])):
        by_task: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
        missing: dict[str, int] = defaultdict(int)
        for r in rs:
            by_task[r["arm"]][r["task"]].append(0.0 if r["reward"] is None else r["reward"])
            missing[r["arm"]] += r["reward"] is None
        arms = sorted(by_task, key=lambda a: (a != baseline, a))
        summary = {a: {"tasks": len(by_task[a]), "trials": sum(map(len, by_task[a].values())),
                       "no_reward": missing[a],
                       "mean": statistics.fmean(statistics.fmean(vs) for vs in by_task[a].values())}
                   for a in arms}
        comparisons = []
        for arm in arms:
            if arm == baseline or baseline not in by_task:
                continue
            tasks = sorted(set(by_task[arm]) & set(by_task[baseline]))
            pairs = [(statistics.fmean(by_task[baseline][t]), statistics.fmean(by_task[arm][t]))
                     for t in tasks]
            deltas = [b - a for a, b in pairs]
            point = statistics.fmean(deltas) if deltas else None
            # As parley-sim's paired tests: under three matched tasks, no interval.
            nan = (float("nan"), float("nan"))
            ci95 = bootstrap_ci(deltas, level=0.95) if len(deltas) >= 3 else nan
            ci90 = bootstrap_ci(deltas, level=0.90) if len(deltas) >= 3 else nan
            _, p = _paired_permutation_p(pairs, statistics.fmean)
            comparisons.append({"arm": arm, "tasks": len(tasks), "delta": point, "ci95": ci95,
                                "ci90": ci90, "p": p, "verdict": _verdict(point, ci95, ci90, margin)})
        out["groups"].append({"agent": agent, "model": model, "user_agent": user_agent,
                              "user_model": user_model, "arms": summary,
                              "comparisons": comparisons})
    return out


def _verdict(point: float | None, ci95: tuple[float, float], ci90: tuple[float, float],
             margin: float | None) -> str:
    """As parley-sim's margin rule: different if the 95% CI excludes 0 and the
    delta reaches the margin; equivalent if the 90% CI lies inside it."""
    if point is None or not all(math.isfinite(x) for x in (*ci95, *ci90)):
        return "-"
    excludes = ci95[0] > 0 or ci95[1] < 0
    if margin is None:
        return "different" if excludes else "-"
    if excludes and abs(point) >= margin:
        return "different"
    if -margin < ci90[0] and ci90[1] < margin:
        return "equivalent"
    return "inconclusive"


def render_sweep(report: dict[str, Any]) -> str:
    def f(x: Any) -> str:
        return "-" if x is None or (isinstance(x, float) and not math.isfinite(x)) else f"{x:+.3f}"

    lines = ["# Persona sweep", "",
             f"Baseline arm: {report['baseline']}. Matched unit: the task (each arm's "
             "reward averaged over its trials of a task; an arm's mean is the mean of its "
             "task means). "
             + (f"Margin: {report['margin']}." if report["margin"] is not None else "No margin.")]
    for g in report["groups"]:
        lines += ["", f"## {g['agent']} / {g['model']}, simulated user {g['user_agent']} / "
                  f"{g['user_model']}", "",
                  "| arm | tasks | trials | no reward | mean reward |", "|---|---|---|---|---|"]
        for arm, s in g["arms"].items():
            lines.append(f"| {arm} | {s['tasks']} | {s['trials']} | {s['no_reward']} | {s['mean']:.3f} |")
        if g["comparisons"]:
            lines += ["", f"| arm vs {report['baseline']} | delta [95% CI] | 90% CI | p | tasks | verdict |",
                      "|---|---|---|---|---|---|"]
            for c in g["comparisons"]:
                lines.append(f"| {c['arm']} | {f(c['delta'])} [{f(c['ci95'][0])}, {f(c['ci95'][1])}] | "
                             f"[{f(c['ci90'][0])}, {f(c['ci90'][1])}] | "
                             f"{'-' if not math.isfinite(c['p']) else round(c['p'], 3)} | "
                             f"{c['tasks']} | {c['verdict']} |")
    return "\n".join(lines) + "\n"
