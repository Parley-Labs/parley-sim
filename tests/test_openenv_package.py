import importlib
import json
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


def test_release_trace_is_checked_at_startup_and_reaches_sessions(monkeypatch):
    from fastapi.testclient import TestClient

    from parley_sim.envs.support_desk import make_policy
    with pytest.raises(SystemExit, match="PARLEY_RELEASE_TRACE"):
        load_app(monkeypatch, PARLEY_RELEASE_TRACE="sometimes")
    mod = load_app(monkeypatch, PARLEY_RELEASE_TRACE="never")
    policy = make_policy("agent_scripted", "agent")
    with TestClient(mod.app) as client, client.websocket_connect("/ws") as ws:
        ws.send_text(json.dumps({"type": "reset", "data": {"seed": 5}}))
        data = ws.receive_json()["data"]
        while not data["done"]:
            move = policy.decide(data["observation"]["observation"])
            ws.send_text(json.dumps({"type": "step", "data": {"move": move}}))
            data = ws.receive_json()["data"]
        ws.send_text(json.dumps({"type": "state"}))
        state = ws.receive_json()["data"]
    assert state["done"] and state["metrics"]
    assert not any(l["type"] in ("header", "footer") for l in state["trace"])
    assert all(l["type"] == "event" or l["seat"] == "agent" for l in state["trace"])


def test_session_timeout_shorter_than_a_model_call_fails_startup(monkeypatch):
    # Builds a real OpenAI-compatible client (no request is sent) and closes it.
    monkeypatch.setenv("OPENROUTER_API_KEY", "placeholder")
    peer = dict(PARLEY_SEATS='{"agent": "client", "customer": "model:peer"}',
                PARLEY_LINEUP='[{"name": "peer", "provider": "openai_compat", "model": "x", '
                              '"base_url": "http://localhost:9/v1", "api_key_env": "OPENROUTER_API_KEY", '
                              '"timeout_s": 60}]')
    with pytest.raises(ValueError, match="PARLEY_SESSION_TIMEOUT"):
        load_app(monkeypatch, **peer, PARLEY_SESSION_TIMEOUT="100")
    assert load_app(monkeypatch, **peer, PARLEY_SESSION_TIMEOUT="120").app is not None


def test_stateless_routes_and_web_interface_are_off_even_when_pushed(monkeypatch):
    from fastapi.testclient import TestClient
    mod = load_app(monkeypatch, ENABLE_WEB_INTERFACE="true")  # what `openenv push` injects
    client = TestClient(mod.app)
    assert client.post("/reset", json={"seed": 1}).status_code == 405
    assert client.get("/web").status_code == 404
