"""Persona sweep over Harbor's simulated-user trials (experimental)."""

import json
import random
from pathlib import Path

import pytest

from parley_sim.adapters import harbor_sweep as hs
from parley_sim.cli import main


def _trial(root: Path, name: str, *, task: str, persona: str | None, reward: float | None,
           agent: str = "claude-code", model: str = "anthropic/claude-sonnet-5-5",
           user_model: str = "anthropic/claude-haiku-5-5", reward_name: str = "reward") -> None:
    t = root / name
    t.mkdir(parents=True)
    (t / "config.json").write_text(json.dumps({
        "task": {"path": f"tasks/{task}"},
        "agent": {"name": agent, "model_name": model},
        "user_agent": {"name": "claude-code", "model_name": user_model,
                       "user_persona_path": persona}}))
    rewards = None if reward is None else {reward_name: reward}
    (t / "result.json").write_text(json.dumps({
        "task_name": task, "verifier_result": None if rewards is None else {"rewards": rewards},
        "exception_info": None if reward is not None else {"exception_type": "AgentTimeoutError"}}))


def test_arm_of_names():
    assert hs.arm_of(None) == ("harbor-default", None)
    assert hs.arm_of("/p/usersim-s07.md") == ("usersim", 7)
    assert hs.arm_of("minimal.md") == ("minimal", None)
    assert hs.arm_of("terse-boss-s1.md") == ("terse-boss", 1)


def test_plan_has_one_run_per_task_and_persona():
    plan = hs.write_plan(["tasks/a", "tasks/b"], [None, "p/usersim-s00.md", "p/minimal.md"],
                         agent="claude-code", model="anthropic/x", user_agent="claude-code",
                         user_model="anthropic/y", jobs="jobs", replicates=3)
    runs = [l for l in plan.splitlines() if l.startswith("harbor run")]
    assert len(runs) == 6
    assert all("--bridge acp" in r and "-k 3" in r for r in runs)
    assert sum("--user-persona-path" in r for r in runs) == 4
    assert "--job-name a__usersim-s00" in plan and "--job-name b__harbor-default" in plan


def test_collect_and_report_find_a_persona_effect(tmp_path: Path):
    rng = random.Random(0)
    tasks = [f"t{i}" for i in range(12)]
    for t in tasks:
        for k in range(3):  # default persona succeeds more often than the terse one
            _trial(tmp_path, f"{t}-d{k}", task=t, persona=None, reward=float(rng.random() < 0.8))
            _trial(tmp_path, f"{t}-m{k}", task=t, persona="p/minimal.md",
                   reward=float(rng.random() < 0.2))
            _trial(tmp_path, f"{t}-u{k}", task=t, persona=f"p/usersim-s{k:02d}.md",
                   reward=float(rng.random() < 0.8))
    _trial(tmp_path, "crashed", task="t0", persona="p/minimal.md", reward=None)
    (tmp_path / "job-result.json").write_text("{}")  # job-level files are skipped
    rows = hs.collect([tmp_path])
    assert len(rows) == 12 * 9 + 1
    report = hs.sweep_report(rows, margin=0.2)
    (g,) = report["groups"]
    assert g["arms"]["minimal"]["no_reward"] == 1 and g["arms"]["minimal"]["trials"] == 37
    comp = {c["arm"]: c for c in g["comparisons"]}
    assert comp["minimal"]["tasks"] == 12 and comp["minimal"]["delta"] < -0.3
    assert comp["minimal"]["verdict"] == "different" and comp["minimal"]["p"] < 0.05
    assert comp["usersim"]["verdict"] in ("equivalent", "inconclusive")
    text = hs.render_sweep(report)
    assert "| minimal |" in text and "arm vs harbor-default" in text


def test_report_needs_three_tasks_for_an_interval(tmp_path: Path):
    for t in ("a", "b"):
        _trial(tmp_path, f"{t}-d", task=t, persona=None, reward=1.0)
        _trial(tmp_path, f"{t}-m", task=t, persona="p/minimal.md", reward=0.0)
    (c,) = hs.sweep_report(hs.collect([tmp_path]), margin=0.1)["groups"][0]["comparisons"]
    assert c["tasks"] == 2 and c["verdict"] == "-"


def test_cli_plan_and_report(tmp_path: Path, capsys):
    personas = tmp_path / "personas"
    personas.mkdir()
    (personas / "minimal.md").write_text(hs.MINIMAL_PERSONA)
    assert main(["harbor-sweep", "plan", "--tasks", "tasks/a", "--personas", str(personas),
                 "--include-default", "--agent", "claude-code", "--model", "m", "--user-agent",
                 "claude-code", "--user-model", "u", "--jobs", "jobs"]) == 0
    assert capsys.readouterr().out.count("harbor run") == 2
    for t in ("a", "b", "c"):
        _trial(tmp_path / "jobs", f"{t}-d", task=t, persona=None, reward=1.0)
        _trial(tmp_path / "jobs", f"{t}-m", task=t, persona="minimal.md", reward=0.0)
    assert main(["harbor-sweep", "report", str(tmp_path / "jobs"), "--margin", "0.2"]) == 0
    assert "| minimal |" in capsys.readouterr().out
    assert main(["harbor-sweep", "report", str(tmp_path / "empty")]) == 1


def test_personas_from_a_panel(tmp_path: Path):
    pytest.importorskip("usersim")
    from test_usersim_persona import PERSONA

    panel = tmp_path / "panel.jsonl"
    panel.write_text("\n".join(json.dumps({"persona": PERSONA}) for _ in range(3)) + "\n")
    paths = hs.write_personas(panel, tmp_path / "out", 2)
    assert [p.name for p in paths] == ["usersim-s00.md", "usersim-s01.md", "minimal.md"]
    text = paths[0].read_text()
    assert text.startswith(hs.INVARIANTS) and "<YOUR_PERSONA>" in text
    meta = json.loads((tmp_path / "out" / "usersim-s00.json").read_text())
    assert meta["interaction_style"] == "confrontational"
    with pytest.raises(ValueError):
        hs.write_personas(panel, tmp_path / "out2", 5)
    # A smaller rerun into the same directory leaves no stale personas; yours stay.
    (tmp_path / "out" / "boss.md").write_text("my own persona")
    hs.write_personas(panel, tmp_path / "out", 1)
    assert sorted(p.name for p in (tmp_path / "out").glob("*.md")) == ["boss.md", "minimal.md",
                                                                      "usersim-s00.md"]


def test_a_requested_reward_key_is_not_replaced_by_another(tmp_path: Path):
    _trial(tmp_path, "t0", task="a", persona=None, reward=0.8, reward_name="score")
    assert hs.collect([tmp_path])[0]["reward"] == 0.8  # default key: a lone reward is used
    assert hs.collect([tmp_path], reward_key="completed")[0]["reward"] is None


def test_user_models_are_not_compared_as_personas(tmp_path: Path):
    for i in range(3):
        _trial(tmp_path, f"d{i}", task=f"t{i}", persona=None, reward=1.0)
        _trial(tmp_path, f"m{i}", task=f"t{i}", persona="minimal.md", reward=0.0,
               user_model="other/model")
    groups = hs.sweep_report(hs.collect([tmp_path]))["groups"]
    assert len(groups) == 2 and all(g["comparisons"] == [] for g in groups)
    assert "simulated user claude-code / other/model" in hs.render_sweep({"baseline": "harbor-default",
                                                                          "margin": None, "groups": groups})


def test_arm_mean_is_the_mean_of_task_means(tmp_path: Path):
    for i in range(10):
        _trial(tmp_path, f"a{i}", task="a", persona=None, reward=1.0)
    _trial(tmp_path, "b0", task="b", persona=None, reward=0.0)
    arm = hs.sweep_report(hs.collect([tmp_path]))["groups"][0]["arms"]["harbor-default"]
    assert arm["mean"] == 0.5 and arm["trials"] == 11 and arm["tasks"] == 2


def test_overlapping_job_dirs_count_each_trial_once(tmp_path: Path):
    _trial(tmp_path / "job", "t0", task="a", persona=None, reward=1.0)
    assert len(hs.collect([tmp_path / "job", tmp_path / "job" / "t0"])) == 1


def test_equivalence_bounds_are_inclusive():
    assert hs._verdict(0.05, (-0.01, 0.125), (0.0, 0.125), 0.125) == "equivalent"


def test_cli_json_report_is_strict_json_for_small_sweeps(tmp_path: Path, capsys):
    from parley_sim.cli import main

    for i in range(2):  # two tasks: no interval
        _trial(tmp_path, f"d{i}", task=f"t{i}", persona=None, reward=1.0)
        _trial(tmp_path, f"m{i}", task=f"t{i}", persona="minimal.md", reward=0.0)
    assert main(["harbor-sweep", "report", str(tmp_path), "--json"]) == 0
    out = capsys.readouterr().out
    report = json.loads(out, parse_constant=lambda c: pytest.fail(f"non-standard JSON {c}"))
    assert report["groups"][0]["comparisons"][0]["ci95"] == [None, None]
