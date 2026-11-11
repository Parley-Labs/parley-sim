import importlib
import sys
from pathlib import Path

import pytest
import yaml

pytest.importorskip("openenv")

PKG = Path(__file__).resolve().parents[1] / "integrations" / "openenv" / "parley_sim_env"


def load_app(monkeypatch, **env):
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    monkeypatch.syspath_prepend(str(PKG.parent))
    for name in [m for m in sys.modules if m.startswith("parley_sim_env")]:
        del sys.modules[name]
    return importlib.import_module("parley_sim_env.server.app")


def test_openenv_manifest_points_at_the_app():
    spec = yaml.safe_load((PKG / "openenv.yaml").read_text())
    assert spec == {"spec_version": 1, "name": "parley_sim", "type": "space",
                    "runtime": "fastapi", "app": "server.app:app", "port": 8000}
    assert (PKG / "server" / "Dockerfile").exists()


def test_packaged_app_reads_config_and_serves_health(monkeypatch):
    from fastapi.testclient import TestClient
    mod = load_app(monkeypatch, PARLEY_SEATS='{"agent": "client", "customer": "policy:conceder"}',
                   PARLEY_ENV_CONFIG='{"max_turns": 6}')
    with TestClient(mod.app) as client:
        assert client.get("/health").status_code == 200
    assert mod._json("PARLEY_SEATS", None)["customer"] == "policy:conceder"


def test_bad_json_config_fails_loudly(monkeypatch):
    with pytest.raises(SystemExit, match="PARLEY_SEATS"):
        load_app(monkeypatch, PARLEY_SEATS="{not json")
