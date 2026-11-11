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
