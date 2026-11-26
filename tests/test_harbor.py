"""Harbor export (experimental): the sidecar runtime and the task generator."""

import asyncio
from pathlib import Path

import pytest

pytest.importorskip("openenv")
pytest.importorskip("mcp")
httpx = pytest.importorskip("httpx")

from parley_sim.adapters.harbor import (  # noqa: E402
    HarborRuntime,
    RuntimeConfig,
    create_runtime_app,
    export_task,
)
from parley_sim.runtime.episode import verify  # noqa: E402

TOKEN = "t" * 24
HEADERS = {"content-type": "application/json", "accept": "application/json, text/event-stream"}
PROPOSE = {"type": "propose", "deal": {"discount_pct": 20, "credit_usd": 25,
                                        "upgrade": "none", "term": "annual"}}


def _play(counterpart: str, moves: list[dict], seed: int = 1060) -> tuple[list[str], dict, int]:
    rt = HarborRuntime(RuntimeConfig(seed=seed, counterpart=counterpart, token=TOKEN, arm="soc"))
    app = create_runtime_app(rt)

    async def go():
        texts: list[str] = []
        async with app.router.lifespan_context(app):
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(transport=transport, base_url="http://parley-runtime:8000") as c:
                async def tool(name, **args):
                    r = await c.post("/mcp", headers=HEADERS, json={
                        "jsonrpc": "2.0", "id": 1, "method": "tools/call",
                        "params": {"name": name, "arguments": args}})
                    r.raise_for_status()
                    return r.json()["result"]["content"][0]["text"]

                texts.append(await tool("start_negotiation"))
                for move in moves:
                    texts.append(await tool("take_turn", message="hi", move=move, explanation="x"))
                    if "is over" in texts[-1]:
                        break
                forbidden = (await c.get("/result")).status_code
                result = (await c.get("/result", headers={"x-parley-token": TOKEN})).json()
        return texts, result, forbidden

    return asyncio.run(go())


def test_episode_plays_and_scores():
    texts, result, forbidden = _play("policy:conceder", [PROPOSE] * 10)
    assert forbidden == 403
    assert "Your private information" in texts[0]
    assert "is over" in texts[-1]
    assert result["completed"] == 1 and result["verified"] == 1 and result["problems"] == []
    lines = result["trace"]
    assert lines[0]["type"] == "header" and lines[-1]["type"] == "footer"
    from parley_sim import envs
    entry = envs.get("support_desk")
    assert verify(entry.env(), lines, make_policy=entry.make_policy) == []
    metrics = entry.episode_metrics(lines)
    assert result["no_deal"] == int(metrics["deal"] == 0.0)
    assert result["reward"] == pytest.approx(metrics["focal_surplus_share"])


def test_walk_away_is_no_deal_with_zero_reward():
    texts, result, _ = _play("policy:hardliner", [{"type": "walk_away"}])
    assert "is over" in texts[-1]
    assert result == {**result, "completed": 1, "no_deal": 1, "reward": 0.0, "verified": 1}


def test_unfinished_episode_scores_zero_and_hides_counterpart():
    _, result, _ = _play("policy:hardliner", [])
    assert result["completed"] == 0 and result["reward"] == 0.0
    # Mid-episode the trace is the focal seat's view only: no counterpart decisions.
    assert all(l["type"] == "event" or l.get("seat") == "agent" for l in result["trace"])


def test_bad_move_is_a_strike_not_an_error():
    texts, result, _ = _play("policy:hardliner", [{"type": "fly"}, {"type": "walk_away"}])
    assert "could not be used" in texts[1]
    assert result["completed"] == 1 and result["verified"] == 1


def test_result_needs_a_token():
    with pytest.raises(ValueError):
        RuntimeConfig.from_env({"PARLEY_SEED": "1", "PARLEY_COUNTERPART": "policy:hardliner"})


def test_export_writes_a_task(tmp_path: Path):
    wheel = tmp_path / "parley_sim-0.0.1-py3-none-any.whl"
    wheel.write_bytes(b"")
    task = export_task(tmp_path / "out", seed=1060, counterpart="policy:hardliner", wheel=wheel)
    assert task.name == "support-desk__s1060-scripted-hardliner"
    for f in ("task.toml", "instruction.md", "environment/Dockerfile", "environment/docker-compose.yaml",
              "environment/runtime/Dockerfile", "tests/test.sh", "solution/solve.sh"):
        assert (task / f).is_file(), f
    toml = (task / "task.toml").read_text()
    compose = (task / "environment" / "docker-compose.yaml").read_text()
    token = toml.split('PARLEY_RESULT_TOKEN = "')[1].split('"')[0]
    assert len(token) >= 16 and f"PARLEY_RESULT_TOKEN={token}" in compose
    # Nothing the agent's container sees names the seed, counterpart or token.
    seen = (task / "instruction.md").read_text() + (task / "environment" / "Dockerfile").read_text()
    assert "1060" not in seen and "hardliner" not in seen and token not in seen
    assert not (task / "environment" / "extra-ca.crt").exists()
    assert "{name}" not in toml and "{service}" not in compose  # every placeholder filled
