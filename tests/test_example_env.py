import asyncio
from pathlib import Path

from parley_sim.cli import main
from parley_sim.matched import load_config, run_matched
from parley_sim.metrics import build_report, render

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"


def test_haggle_example_runs_reports_and_replays(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(EXAMPLES))
    asyncio.run(run_matched(load_config(EXAMPLES / "haggle.yaml"), tmp_path))
    report = build_report(tmp_path)
    assert report["cells"] == {"done": 30}
    text = render(report)
    assert "focal_surplus_share" in text and "Acceptance by offer quality" not in text
    assert "p=nan" not in text
    traces = sorted(str(p) for p in (tmp_path / "traces").glob("*.jsonl"))
    assert main(["verify", *traces, "--env", "haggle:ENTRY"]) == 0


def test_haggle_report_in_a_fresh_process_needs_env(tmp_path, monkeypatch, capsys):
    """What `parley-sim report` sees in a new process: the manifest names the
    import path, which is not imported unless the user passes --env."""
    from parley_sim import envs
    monkeypatch.syspath_prepend(str(EXAMPLES))
    asyncio.run(run_matched(load_config(EXAMPLES / "haggle.yaml"), tmp_path))
    traces = sorted(str(p) for p in (tmp_path / "traces").glob("*.jsonl"))
    monkeypatch.setattr(envs, "_registered", {})  # forget what this process loaded
    assert main(["report", str(tmp_path)]) == 2
    assert "--env haggle:ENTRY" in capsys.readouterr().err
    assert main(["verify", traces[0]]) == 1  # the header's "haggle" is not known either
    assert main(["report", str(tmp_path), "--env", "support_desk"]) == 2
    assert "is a trace of env 'haggle', not 'support_desk'" in capsys.readouterr().err
    assert main(["report", str(tmp_path), "--env", "haggle:ENTRY",
                 "--margins", "focal_surplus_share=0.1"]) == 0
    assert "focal_surplus_share" in capsys.readouterr().out
    assert main(["verify", *traces, "--env", "haggle:ENTRY"]) == 0


def test_readme_config_is_valid(tmp_path):
    readme = (Path(__file__).resolve().parents[1] / "README.md").read_text()
    block = readme.split("```yaml\n", 1)[1].split("```", 1)[0]
    (tmp_path / "c.yaml").write_text(block)
    cfg = load_config(tmp_path / "c.yaml")
    assert cfg.focal_models == ["agent-model"]
