import asyncio

import pytest

from parley_sim import envs
from parley_sim.cli import main, smoke_config
from parley_sim.envs.support_desk import SupportDesk
from parley_sim.matched import MatchedConfig, run_matched
from parley_sim.metrics import build_report


class RenamedDesk(SupportDesk):
    name = "renamed_desk"


ENTRY = envs.EnvEntry(**{**envs.get("support_desk").__dict__, "env": RenamedDesk})


def test_env_by_import_path_runs_reports_and_verifies(tmp_path):
    cfg = MatchedConfig(**{**smoke_config("0:2").model_dump(),
                           "env": "test_env_plugin:ENTRY"})
    asyncio.run(run_matched(cfg, tmp_path))
    report = build_report(tmp_path)
    assert report["cells"]["done"] > 0
    traces = sorted(str(p) for p in (tmp_path / "traces").glob("*.jsonl"))
    assert main(["verify", *traces]) == 0


def test_register_and_unknown_env():
    envs.register("my_desk", ENTRY)
    assert envs.get("my_desk") is ENTRY
    with pytest.raises(TypeError):
        envs.register("bad", object())
    with pytest.raises(ValueError, match="import path"):
        envs.get("no_such_env")


SIDE_EFFECT_MODULE = '''
import os, pathlib
pathlib.Path(os.environ["PARLEY_TEST_MARK"]).write_text("imported")
ENTRY = None
'''


def test_import_paths_named_by_data_files_are_never_imported(tmp_path, monkeypatch, capsys):
    import importlib
    import json
    import sys
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "sidefx_mod.py").write_text(SIDE_EFFECT_MODULE)
    monkeypatch.syspath_prepend(str(tmp_path / "pkg"))
    mark = tmp_path / "imported.txt"
    monkeypatch.setenv("PARLEY_TEST_MARK", str(mark))
    imported = []
    real_import = importlib.import_module
    monkeypatch.setattr(importlib, "import_module",
                        lambda name, *a: imported.append(name) or real_import(name, *a))

    run = tmp_path / "run"
    asyncio.run(run_matched(smoke_config("0:1"), run))
    trace = sorted((run / "traces").glob("*.jsonl"))[0]
    lines = trace.read_text().splitlines()
    header = json.loads(lines[0])
    header["env"]["name"] = "sidefx_mod:ENTRY"
    trace.write_text("\n".join([json.dumps(header), *lines[1:]]) + "\n")
    manifest = run / "manifest.jsonl"
    rows = [json.loads(l) for l in manifest.read_text().splitlines()]
    rows[0]["config"]["env"] = "sidefx_mod:ENTRY"
    manifest.write_text("".join(json.dumps(r) + "\n" for r in rows))

    assert main(["verify", str(trace)]) == 1
    assert "only loaded when you name it yourself" in capsys.readouterr().out
    assert main(["report", str(run)]) == 2
    assert "--env sidefx_mod:ENTRY" in capsys.readouterr().err
    assert main(["claims", str(run), "--prefix", str(tmp_path / "c")]) == 2
    with pytest.raises(ValueError, match="import path"):
        envs.get("sidefx_mod:ENTRY")
    assert not mark.exists() and "sidefx_mod" not in sys.modules
    assert "sidefx_mod" not in imported


class ShadowDesk(SupportDesk):
    pass  # keeps SupportDesk's name, "support_desk"


SHADOW = envs.EnvEntry(**{**envs.get("support_desk").__dict__, "env": ShadowDesk})
NOT_AN_ENTRY = 42


def needs_args(x):
    return ENTRY


def returns_wrong():
    return "support_desk"


def make_entry():
    return ENTRY


def test_import_path_cannot_replace_a_builtin_or_registered_env(monkeypatch):
    monkeypatch.setattr(envs, "_registered", {})
    builtin = envs.get("support_desk")
    with pytest.raises(ValueError, match="built-in"):
        envs.get("test_env_plugin:SHADOW", allow_import=True)
    assert envs.get("support_desk") == builtin
    assert "test_env_plugin:SHADOW" not in envs.registry()
    envs.register("renamed_desk", envs.EnvEntry(**{**ENTRY.__dict__, "metric_kinds": {}}))
    with pytest.raises(ValueError, match="already registered"):
        envs.get("test_env_plugin:ENTRY", allow_import=True)


def test_import_path_must_give_an_env_entry(monkeypatch):
    monkeypatch.setattr(envs, "_registered", {})
    for name in ("NOT_AN_ENTRY", "needs_args", "returns_wrong", "ShadowDesk"):
        with pytest.raises(TypeError):
            envs.get(f"test_env_plugin:{name}", allow_import=True)
    with pytest.raises(ValueError, match="no attribute"):
        envs.get("test_env_plugin:MISSING", allow_import=True)
    assert envs.get("test_env_plugin:make_entry", allow_import=True) is ENTRY
    assert envs.get("renamed_desk") is ENTRY  # traces name the env by its own name
