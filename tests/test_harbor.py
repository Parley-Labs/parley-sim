"""Harbor export (experimental): sidecar runtime, task export and collection."""

import asyncio
import json
from pathlib import Path

import pytest

pytest.importorskip("mcp")
httpx = pytest.importorskip("httpx")

from parley_sim import envs  # noqa: E402
from parley_sim.adapters.harbor import (  # noqa: E402
    HarborCell,
    HarborRuntime,
    RuntimeConfig,
    collect_jobs,
    create_runtime_app,
    export_task,
    harbor_cells,
)
from parley_sim.cli import main  # noqa: E402
from parley_sim.matched import MatchedConfig, build_grid, config_hash, load_config  # noqa: E402
from parley_sim.metrics import build_report  # noqa: E402
from parley_sim.runtime.episode import verify  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
TOKEN = "t" * 24
HEADERS = {"content-type": "application/json", "accept": "application/json, text/event-stream"}
PROPOSE = {"type": "propose", "deal": {"discount_pct": 20, "credit_usd": 25,
                                        "upgrade": "none", "term": "annual"}}
CUSTOMER = {"message": "Not enough.", "move": {"type": "propose", "deal": {
    "discount_pct": 30, "credit_usd": 25, "upgrade": "none", "term": "annual"}}, "explanation": "x"}


class FakeClient:
    """A model counterpart that always makes the same reply, or fails."""

    def __init__(self, reply=CUSTOMER):
        self.reply = reply
        self.calls = 0

    async def complete(self, *, system, user, schema, history=None):
        self.calls += 1
        if isinstance(self.reply, Exception):
            raise self.reply
        return dict(self.reply), {"usage": {"input_tokens": 10, "output_tokens": 5}}

    async def aclose(self):
        pass


def config(**kw) -> MatchedConfig:
    base = dict(run_id="hx", env="support_desk", seeds="1060:1063",
                lineup=[{"name": "focal", "provider": "scripted", "model": "agent_scripted"},
                        {"name": "llm", "provider": "openai_compat", "model": "some/model",
                         "base_url": "https://openrouter.ai/api/v1"}],
                focal_models=["focal"], scripted_counterparts=["hardliner", "reasonable", "conceder"],
                adaptive_counterparts=["llm"], self_null_replicates=1)
    return MatchedConfig(**{**base, **kw})


def cells(cfg: MatchedConfig) -> list[HarborCell]:
    return harbor_cells(build_grid(cfg), run_id=cfg.run_id, env=cfg.env, env_config=cfg.env_config,
                        lineup=cfg.lineup)


def play(cell: HarborCell, moves, client=None):
    """Play `moves` through the sidecar over MCP; returns (texts, result, unauthenticated status)."""
    rt = HarborRuntime(RuntimeConfig(cell=cell, token=TOKEN),
                       client_factory=lambda spec: client or FakeClient())
    app = create_runtime_app(rt)

    async def go():
        texts = []
        async with app.router.lifespan_context(app):
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(transport=transport, base_url="http://parley-runtime:8000") as c:
                async def tool(name, **args):
                    r = await c.post("/mcp", headers=HEADERS, json={
                        "jsonrpc": "2.0", "id": 1, "method": "tools/call",
                        "params": {"name": name, "arguments": args}})
                    r.raise_for_status()
                    return r.json()["result"]["content"][0]["text"]

                texts.append(await tool("start_episode"))
                for move in moves:
                    if "Do not call any more tools" in texts[-1]:
                        break
                    m = move(texts[-1]) if callable(move) else move
                    texts.append(await tool("take_turn", action={"message": "hi", "move": m,
                                                                 "explanation": "x"}))
                forbidden = (await c.get("/result")).status_code
                result = (await c.get("/result", headers={"x-parley-token": TOKEN})).json()
        return texts, result, forbidden

    return asyncio.run(go())


def accept_last(text):
    return {"type": "accept"} if "Turns you have left, including this one: 1" in text else PROPOSE


def test_cells_follow_the_local_grid():
    cfg = load_config(ROOT / "configs" / "equiv-sd3.yaml")
    hc = cells(cfg)
    # 40 seeds x (soc r0 + usersim r0, r1 + aoc r0, r1), no focal model.
    assert len(hc) == 200 and len({c.name for c in hc}) == 200
    grid = {(c.seed, c.arm, c.replicate): c for c in build_grid(cfg) if c.focal == cfg.focal_models[0]}
    for c in hc:
        local = grid[(c.seed, c.arm, c.replicate)]
        assert c.seats == {s: l for s, l in local.seats.items() if s != local.focal_seat}
        personas = {m.get("persona") for m in c.lineup}
        assert personas == {"soc": set(), "aoc": {None}, "usersim": {"usersim"}}[c.arm]
    assert HarborCell.decode(hc[0].encode()) == hc[0]


def test_scripted_episode_plays_and_scores():
    soc = next(c for c in cells(config()) if c.arm == "soc" and "conceder" in str(c.seats))
    texts, result, forbidden = play(soc, [accept_last] * 10)
    assert forbidden == 403
    assert "Your private information" in texts[0]
    assert "Respond with a JSON object" not in texts[0] and "take_turn" in texts[0]
    assert result["completed"] == 1 and result["verified"] == 1 and result["problems"] == []
    entry = envs.get("support_desk")
    lines = result["trace"]
    assert verify(entry.env(), lines, make_policy=entry.make_policy) == []
    metrics = entry.episode_metrics(lines)
    assert result["no_deal"] == int(metrics["deal"] == 0.0)
    assert result["reward"] == pytest.approx(metrics["focal_surplus_share"])
    assert lines[0]["spec"]["arm"] == "soc" and lines[0]["spec"]["seed"] == soc.seed


def test_model_counterpart_through_fake_client():
    aoc = next(c for c in cells(config()) if c.arm == "aoc")
    client = FakeClient()
    _, result, _ = play(aoc, [accept_last] * 10, client=client)
    assert client.calls > 0 and result["completed"] == 1 and result["verified"] == 1
    usage = [l["raw"]["usage"] for l in result["trace"]
             if l["type"] == "decision" and l["seat"] == "customer"]
    assert usage and all(u["input_tokens"] == 10 for u in usage)


def test_failing_counterpart_ends_the_episode_as_errored():
    aoc = next(c for c in cells(config()) if c.arm == "aoc")
    texts, result, _ = play(aoc, [PROPOSE] * 10, client=FakeClient(RuntimeError("provider down")))
    assert "ended early" in texts[-1]
    assert result["errored"] == 1 and result["completed"] == 0 and result["reward"] == 0.0


def test_walk_away_is_no_deal_with_zero_reward():
    soc = next(c for c in cells(config()) if c.arm == "soc")
    texts, result, _ = play(soc, [{"type": "walk_away"}])
    assert "is over" in texts[-1]
    assert (result["completed"], result["no_deal"], result["reward"], result["verified"]) == (1, 1, 0.0, 1)


def test_unfinished_episode_scores_zero_and_hides_the_counterpart():
    aoc = next(c for c in cells(config()) if c.arm == "aoc")
    _, result, _ = play(aoc, [])
    assert result["completed"] == 0 and result["reward"] == 0.0
    assert all(l["type"] == "event" or l.get("seat") == aoc.focal_seat for l in result["trace"])
    assert all("context" not in l for l in result["trace"])


def test_bad_move_is_a_strike_not_an_error():
    soc = next(c for c in cells(config()) if c.arm == "soc")
    texts, result, _ = play(soc, [{"type": "fly"}, {"type": "walk_away"}])
    assert "could not be used" in texts[1]
    assert result["completed"] == 1 and result["verified"] == 1


def test_seed_secret_changes_the_scenario_but_still_verifies():
    soc = next(c for c in cells(config()) if c.arm == "soc")
    rt = HarborRuntime(RuntimeConfig(cell=soc, token=TOKEN, seed_secret="s" * 32))
    asyncio.run(rt.start())
    asyncio.run(rt.take_turn({"message": "bye", "move": {"type": "walk_away"}, "explanation": "x"}))
    result = rt.result()
    assert result["trace"][0]["spec"]["seed"] != soc.seed and result["verified"] == 1


def test_result_needs_a_token():
    with pytest.raises(ValueError):
        RuntimeConfig.from_env({"PARLEY_CELL": "e30="})


def _wheel(tmp_path: Path) -> Path:
    wheel = tmp_path / "parley_sim-0.0.1-py3-none-any.whl"
    wheel.write_bytes(b"")
    return wheel


def test_export_writes_tasks_that_hide_the_cell(tmp_path: Path):
    cfg = config()
    tasks = [export_task(tmp_path / "out", c, chash=config_hash(cfg), wheel=_wheel(tmp_path))
             for c in cells(cfg)]
    assert len(tasks) == 3 * 3  # 3 seeds x (soc r0, aoc r0, aoc r1)
    for task in tasks:
        for f in ("task.toml", "instruction.md", "environment/Dockerfile",
                  "environment/docker-compose.yaml", "environment/runtime/Dockerfile",
                  "tests/test.sh", "solution/solve.sh"):
            assert (task / f).is_file(), f
        toml = (task / "task.toml").read_text()
        compose = (task / "environment" / "docker-compose.yaml").read_text()
        token = toml.split('PARLEY_RESULT_TOKEN = "')[1].split('"')[0]
        assert len(token) >= 16 and f'PARLEY_RESULT_TOKEN: "{token}"' in compose
        # Nothing in the agent's container names the seed, counterpart or token.
        seen = (task / "instruction.md").read_text() + (task / "environment" / "Dockerfile").read_text()
        assert not any(x in seen for x in ("106", "hardliner", "conceder", "some/model", token))
        assert "{" not in toml.split("[task]")[0]  # every placeholder filled
    aoc = (tmp_path / "out" / "support-desk__s1060-aoc-r0" / "environment")
    assert 'OPENROUTER_API_KEY: "${OPENROUTER_API_KEY:-}"' in (aoc / "docker-compose.yaml").read_text()
    assert "[harbor,openai]" in (aoc / "runtime" / "Dockerfile").read_text()


def test_export_persona_arm_mounts_the_users_panel(tmp_path: Path):
    cfg = config(lineup=[*config().lineup,
                         {"name": "llm-p", "provider": "openai_compat", "model": "some/model",
                          "base_url": "https://openrouter.ai/api/v1", "persona": "usersim"}],
                 extra_arms={"usersim": ["llm-p"]}, persona_file="panel.parquet")
    cell = next(c for c in cells(cfg) if c.arm == "usersim")
    task = export_task(tmp_path / "out", cell, chash=config_hash(cfg), image="parley-runtime:test",
                       seed_secret="x" * 32)
    compose = (task / "environment" / "docker-compose.yaml").read_text()
    assert "image: parley-runtime:test" in compose and "build:" not in compose
    assert "${PARLEY_PERSONA_PANEL:?" in compose and "panel.parquet" not in compose
    assert 'PARLEY_SEED_SECRET: "' + "x" * 32 in compose
    assert not (task / "environment" / "runtime").exists()


def _fake_trial(job: Path, name: str, agent: dict, result: dict, started: str) -> None:
    t = job / name
    (t / "verifier").mkdir(parents=True)
    (t / "config.json").write_text(json.dumps({"agent": agent}))
    (t / "result.json").write_text(json.dumps({"started_at": started}))
    trace = result.pop("trace")
    (t / "verifier" / "trace.jsonl").write_text("".join(json.dumps(l) + "\n" for l in trace))
    (t / "verifier" / "result.json").write_text(json.dumps(result))


def test_collect_builds_a_run_that_report_reads(tmp_path: Path):
    cfg = config()
    job = tmp_path / "job"
    agents = [{"name": "claude-code", "model_name": "anthropic/claude-x"},
              {"name": "terminus-2", "model_name": "openrouter/some-model"}]
    for i, cell in enumerate(cells(cfg)):
        for a, agent in enumerate(agents):
            _, result, _ = play(cell, [accept_last] * 10)
            _fake_trial(job, f"t{i}-{a}", agent, result, f"2026-11-26T00:{i:02d}:0{a}")
    # A second attempt of one task is ignored; an unfinished one is kept as errored.
    _, again, _ = play(cells(cfg)[0], [accept_last] * 10)
    _fake_trial(job, "dup", agents[0], again, "2026-11-26T01:00:00")
    counts = collect_jobs(cfg.model_dump(), config_hash(cfg), [job], tmp_path / "run")
    assert counts == {"trials": 19, "kept": 18, "duplicates": 1, "other_config": 0}
    manifest = [json.loads(l) for l in (tmp_path / "run" / "manifest.jsonl").read_text().splitlines()]
    assert manifest[0]["config"]["focal_models"] == ["claude-code.anthropic-claude-x",
                                                     "terminus-2.openrouter-some-model"]
    report = build_report(tmp_path / "run")
    assert report["cells"]["done"] == 18
    assert main(["verify", *map(str, (tmp_path / "run" / "traces").glob("*.jsonl"))]) == 0


def test_cli_export_and_collect(tmp_path: Path, capsys):
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(json.dumps(config().model_dump(exclude_none=True)))  # JSON is YAML
    assert main(["harbor-export", str(cfg_path), "--out", str(tmp_path / "tasks"),
                 "--wheel", str(_wheel(tmp_path))]) == 0
    assert "9 tasks" in capsys.readouterr().out
    assert main(["harbor-export", str(cfg_path), "--out", str(tmp_path / "t2"),
                 "--wheel", str(tmp_path / "nope.whl")]) == 2
    assert main(["harbor-collect", str(tmp_path / "empty"), "--config", str(cfg_path),
                 "--out", str(tmp_path / "run")]) == 1


def test_persona_counterpart_plays_from_the_mounted_panel(tmp_path: Path):
    pytest.importorskip("usersim")
    from test_usersim_persona import PERSONA

    panel = tmp_path / "panel.jsonl"
    panel.write_text(json.dumps({"persona": PERSONA}) + "\n")
    cfg = config(lineup=[*config().lineup,
                         {"name": "llm-p", "provider": "openai_compat", "model": "some/model",
                          "base_url": "https://openrouter.ai/api/v1", "persona": "usersim"}],
                 extra_arms={"usersim": ["llm-p"]}, persona_file=str(panel))
    cell = next(c for c in cells(cfg) if c.arm == "usersim")
    client = FakeClient()
    rt = HarborRuntime(RuntimeConfig(cell=cell, token=TOKEN, persona_file=str(panel)),
                       client_factory=lambda spec: client)
    asyncio.run(rt.start())
    asyncio.run(rt.take_turn({"message": "bye", "move": {"type": "walk_away"}, "explanation": "x"}))
    result = rt.result()
    assert result["completed"] == 1 and result["verified"] == 1 and result["persona_digest"]
    with pytest.raises(ValueError, match="PARLEY_PERSONA_FILE"):
        HarborRuntime(RuntimeConfig(cell=cell, token=TOKEN))


def test_take_turn_advertises_the_env_action_schema():
    soc = next(c for c in cells(config()) if c.arm == "soc")
    app = create_runtime_app(HarborRuntime(RuntimeConfig(cell=soc, token=TOKEN)))

    async def go():
        async with app.router.lifespan_context(app):
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(transport=transport, base_url="http://parley-runtime:8000") as c:
                r = await c.post("/mcp", headers=HEADERS, json={"jsonrpc": "2.0", "id": 1,
                                                                "method": "tools/list", "params": {}})
                return {t["name"]: t for t in r.json()["result"]["tools"]}

    tools = asyncio.run(go())
    assert set(tools) == {"start_episode", "take_turn"}
    action = tools["take_turn"]["inputSchema"]["properties"]["action"]
    schema = envs.get("support_desk").text().action_schema
    assert action["properties"] == schema["properties"] and action["required"] == schema["required"]


def test_reward_metric_is_configurable_and_checked():
    cfg = config()
    joint = harbor_cells(build_grid(cfg), run_id=cfg.run_id, env=cfg.env, env_config={},
                         lineup=cfg.lineup, reward_metric="joint_efficiency")
    soc = next(c for c in joint if c.arm == "soc")
    _, result, _ = play(soc, [accept_last] * 10)
    assert result["reward"] == pytest.approx(result["metrics"]["joint_efficiency"] or 0.0)
    with pytest.raises(ValueError, match="reward metric"):
        harbor_cells(build_grid(cfg), run_id=cfg.run_id, env=cfg.env, env_config={},
                     lineup=cfg.lineup, reward_metric="nope")


def test_oracle_plays_any_env_to_the_end(tmp_path: Path):
    """The generated solve.sh submits the smallest schema-valid action each turn."""
    import base64
    import re

    soc = next(c for c in cells(config()) if c.arm == "soc")
    task = export_task(tmp_path, soc, chash="x", wheel=_wheel(tmp_path))
    solve = (task / "solution" / "solve.sh").read_text()
    schema = json.loads(base64.b64decode(re.search(r'b64decode\("([^"]+)"\)', solve).group(1)))
    assert schema == envs.get("support_desk").text().action_schema
    ns: dict = {}
    exec(solve[solve.index("def minimal"):solve.index('call("initialize"')], ns)
    action = ns["minimal"](schema)
    rt = HarborRuntime(RuntimeConfig(cell=soc, token=TOKEN))
    text = asyncio.run(rt.start())
    for _ in range(50):
        if "Do not call any more tools" in text:
            break
        text = asyncio.run(rt.take_turn(action))
    result = rt.result()
    assert result["completed"] == 1 and result["verified"] == 1
    assert all(l.get("strike") is None for l in result["trace"] if l["type"] == "decision")
