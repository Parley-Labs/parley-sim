"""Harbor export (experimental): sidecar runtime, task export and collection."""

import asyncio
import json
from dataclasses import replace
from pathlib import Path

import pytest

pytest.importorskip("mcp")
httpx = pytest.importorskip("httpx")

from parley_sim import envs  # noqa: E402
from parley_sim.adapters.harbor import (  # noqa: E402
    CHAT_NOTE,
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
    """The config's cells as export_task writes them (stamped with its hash)."""
    chash = config_hash(cfg)
    return [replace(c, config_hash=chash)
            for c in harbor_cells(build_grid(cfg), run_id=cfg.run_id, env=cfg.env,
                                  env_config=cfg.env_config, lineup=cfg.lineup)]


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
                       seed_secret_env="MY_SEED_SECRET")
    compose = (task / "environment" / "docker-compose.yaml").read_text()
    assert "image: parley-runtime:test" in compose and "build:" not in compose
    assert "${PARLEY_PERSONA_PANEL:?" in compose and "panel.parquet" not in compose
    assert 'PARLEY_SEED_SECRET: "${MY_SEED_SECRET:?' in compose  # passed through, never written
    with pytest.raises(ValueError, match="not an environment variable"):
        export_task(tmp_path / "out2", cell, chash=config_hash(cfg), image="x", seed_secret_env='a"b')
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
    assert counts == {"trials": 19, "kept": 18, "duplicates": 1, "other_config": 0, "no_config_hash": 0}
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


def test_collect_rejects_trials_of_another_config(tmp_path: Path):
    """Same run_id, different counterpart model: the old trial must not be
    collected under the new config, before or instead of its own trials."""
    old = config()
    new = config(lineup=[old.lineup[0], {**old.lineup[1].model_dump(exclude_none=True),
                                          "model": "other/model"}])
    assert old.run_id == new.run_id and config_hash(old) != config_hash(new)
    agent = {"name": "claude-code", "model_name": "anthropic/claude-x"}
    task = export_task(tmp_path / "tasks", cells(old)[0], chash=config_hash(old), wheel=_wheel(tmp_path))
    stamped = HarborCell.decode(
        (task / "environment" / "docker-compose.yaml").read_text().split('PARLEY_CELL: "')[1].split('"')[0])
    assert stamped.config_hash == config_hash(old)
    _, result, _ = play(stamped, [accept_last] * 10)
    assert result["cell"]["config_hash"] == config_hash(old)
    _fake_trial(tmp_path / "job", "old", agent, result, "2026-11-27T00:00:00")
    counts = collect_jobs(new.model_dump(), config_hash(new), [tmp_path / "job"], tmp_path / "run")
    assert counts["kept"] == 0 and counts["other_config"] == 1
    counts = collect_jobs(old.model_dump(), config_hash(old), [tmp_path / "job"], tmp_path / "run2")
    assert counts["kept"] == 1
    # A trial from before the hash was recorded is not collected either.
    _, legacy, _ = play(replace(cells(old)[1], config_hash=None), [accept_last] * 10)
    _fake_trial(tmp_path / "job2", "legacy", agent, legacy, "2026-11-27T00:00:00")
    counts = collect_jobs(old.model_dump(), config_hash(old), [tmp_path / "job2"], tmp_path / "run3")
    assert counts["kept"] == 0 and counts["no_config_hash"] == 1


def test_sidecar_loads_an_env_named_by_import_path(monkeypatch):
    """A fresh sidecar process has nothing registered: the cell's import path
    (here the haggle example) must be imported, not rejected."""
    from parley_sim import envs as envs_mod

    examples = ROOT / "examples"
    monkeypatch.syspath_prepend(str(examples))
    cfg = load_config(examples / "haggle.yaml")
    cell = next(c for c in harbor_cells(build_grid(cfg), run_id=cfg.run_id, env=cfg.env,
                                        env_config=cfg.env_config, lineup=cfg.lineup)
                if c.arm == "soc")
    monkeypatch.setattr(envs_mod, "_registered", {})  # as in a new process
    rt = HarborRuntime(RuntimeConfig(cell=HarborCell.decode(cell.encode()), token=TOKEN))
    text = asyncio.run(rt.start())
    assert "take_turn" in text


def test_cancelled_counterpart_turn_ends_the_episode_cleanly():
    """A tool call cancelled while a counterpart decides must not leave the
    consumed focal request pending (every later turn would then fail)."""

    class Hanging(FakeClient):
        async def complete(self, **kw):
            await asyncio.Event().wait()

    aoc = next(c for c in cells(config()) if c.arm == "aoc")
    rt = HarborRuntime(RuntimeConfig(cell=aoc, token=TOKEN), client_factory=lambda spec: Hanging())

    async def go():
        await rt.start()  # the first turn may be the customer's: start it without hanging
        turn = asyncio.create_task(rt.take_turn(
            {"message": "hi", "move": PROPOSE, "explanation": "x"}))
        await asyncio.sleep(0.05)
        turn.cancel()
        with pytest.raises(asyncio.CancelledError):
            await turn
        return await rt.take_turn({"message": "hi", "move": PROPOSE, "explanation": "x"})

    rt_start = rt.start

    async def start_without_hanging():
        rt.clients = {k: FakeClient() for k in rt.clients}  # opening turn answers
        out = await rt_start()
        for seat in rt.counterparts.values():
            seat.client = Hanging()
        return out

    rt.start = start_without_hanging
    text = asyncio.run(go())
    assert "ended early" in text
    result = rt.result()
    assert result["errored"] == 1 and result["completed"] == 0 and rt.pending is None


def test_collect_refuses_mixed_persona_panels(tmp_path: Path):
    cfg = config()
    agent = {"name": "claude-code", "model_name": "anthropic/claude-x"}
    for i, (cell, digest) in enumerate([(cells(cfg)[0], "panel-a"), (cells(cfg)[1], "panel-b")]):
        _, result, _ = play(cell, [accept_last] * 10)
        result["persona_digest"] = digest
        _fake_trial(tmp_path / "job", f"t{i}", agent, result, f"2026-11-27T00:0{i}:00")
    with pytest.raises(ValueError, match="different persona panel"):
        collect_jobs(cfg.model_dump(), config_hash(cfg), [tmp_path / "job"], tmp_path / "run")


def _nested_schema():
    from typing import Literal

    from pydantic import BaseModel

    class Move(BaseModel):
        type: Literal["a", "b"]

    class Action(BaseModel):
        message: str
        move: Move
        backup: Move | None = None

    return Action.model_json_schema()


def test_inline_refs():
    from parley_sim.adapters.harbor import inline_refs

    schema = _nested_schema()
    assert "$defs" in schema
    flat = inline_refs(schema)
    assert "$defs" not in json.dumps(flat) and "$ref" not in json.dumps(flat)
    assert flat["properties"]["move"]["properties"]["type"]["enum"] == ["a", "b"]
    old_style = {"definitions": {"M": {"type": "integer"}}, "properties": {"m": {"$ref": "#/definitions/M"}}}
    assert inline_refs(old_style) == {"properties": {"m": {"type": "integer"}}}
    with pytest.raises(ValueError, match="recursive"):
        inline_refs({"$defs": {"N": {"properties": {"next": {"$ref": "#/$defs/N"}}}},
                     "properties": {"n": {"$ref": "#/$defs/N"}}})
    with pytest.raises(ValueError, match="unresolvable"):
        inline_refs({"properties": {"x": {"$ref": "#/$defs/Missing"}}})


def _tools(rt):
    app = create_runtime_app(rt)

    async def go():
        async with app.router.lifespan_context(app):
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(transport=transport, base_url="http://parley-runtime:8000") as c:
                r = await c.post("/mcp", headers=HEADERS, json={"jsonrpc": "2.0", "id": 1,
                                                                "method": "tools/list", "params": {}})
                return {t["name"]: t for t in r.json()["result"]["tools"]}

    return asyncio.run(go())


def test_nested_action_schema_registers_inlined():
    """A pydantic-generated schema with $defs/$ref used to crash tool
    registration (KeyError '#/$defs/Move')."""
    soc = next(c for c in cells(config()) if c.arm == "soc")
    rt = HarborRuntime(RuntimeConfig(cell=soc, token=TOKEN))
    rt.text = type("T", (), {"action_schema": _nested_schema()})()
    action = _tools(rt)["take_turn"]["inputSchema"]["properties"]["action"]
    assert action["properties"]["move"]["properties"]["type"]["enum"] == ["a", "b"]
    assert "$ref" not in json.dumps(action)


def test_recursive_action_schema_still_starts():
    soc = next(c for c in cells(config()) if c.arm == "soc")
    rt = HarborRuntime(RuntimeConfig(cell=soc, token=TOKEN))
    recursive = {"$defs": {"N": {"type": "object", "properties": {"next": {"$ref": "#/$defs/N"}}}},
                 "type": "object", "properties": {"n": {"$ref": "#/$defs/N"}}}
    rt.text = type("T", (), {"action_schema": recursive})()
    tool = _tools(rt)["take_turn"]
    assert tool["inputSchema"]["properties"]["action"]["type"] == "object"
    assert "#/$defs/N" in tool["description"]  # the full schema is still described


def test_failed_collect_leaves_the_previous_run_intact(tmp_path: Path):
    cfg = config()
    agent = {"name": "claude-code", "model_name": "anthropic/claude-x"}
    for i, cell in enumerate(cells(cfg)[:2]):
        _, result, _ = play(cell, [accept_last] * 10)
        _fake_trial(tmp_path / "good", f"t{i}", agent, result, f"2026-11-27T00:0{i}:00")
    out = tmp_path / "run"
    assert collect_jobs(cfg.model_dump(), config_hash(cfg), [tmp_path / "good"], out)["kept"] == 2
    (out / "notes.txt").write_text("mine")
    before = {p.relative_to(out): p.read_bytes() for p in out.rglob("*") if p.is_file()}
    # A later collect that fails part-way (mixed panels) must change nothing.
    for i, (cell, digest) in enumerate([(cells(cfg)[2], "panel-a"), (cells(cfg)[3], "panel-b")]):
        _, result, _ = play(cell, [accept_last] * 10)
        result["persona_digest"] = digest
        _fake_trial(tmp_path / "bad", f"b{i}", agent, result, f"2026-11-27T01:0{i}:00")
    with pytest.raises(ValueError):
        collect_jobs(cfg.model_dump(), config_hash(cfg), [tmp_path / "good", tmp_path / "bad"], out)
    after = {p.relative_to(out): p.read_bytes() for p in out.rglob("*") if p.is_file()}
    assert after == before
    # A successful re-collect of fewer trials leaves no stale traces behind.
    assert collect_jobs(cfg.model_dump(), config_hash(cfg), [tmp_path / "good" / "t0"], out)["kept"] == 1
    assert len(list((out / "traces").glob("*.jsonl"))) == 1 and (out / "notes.txt").exists()
    assert not (out / "traces.collecting").exists() and not (out / "traces.previous").exists()


def test_collect_publish_failure_restores_the_previous_run(tmp_path: Path, monkeypatch):
    cfg = config()
    agent = {"name": "claude-code", "model_name": "anthropic/claude-x"}
    for i, cell in enumerate(cells(cfg)[:2]):
        _, result, _ = play(cell, [accept_last] * 10)
        _fake_trial(tmp_path / "job", f"t{i}", agent, result, f"2026-11-27T00:0{i}:00")
    out = tmp_path / "run"
    collect_jobs(cfg.model_dump(), config_hash(cfg), [tmp_path / "job" / "t0"], out)
    before = {p.relative_to(out): p.read_bytes() for p in out.rglob("*") if p.is_file()}
    import parley_sim.adapters.harbor as hb

    def fail(*a, **kw):
        raise OSError("disk full")

    monkeypatch.setattr(hb.os, "replace", fail)
    with pytest.raises(OSError):
        collect_jobs(cfg.model_dump(), config_hash(cfg), [tmp_path / "job"], out)
    after = {p.relative_to(out): p.read_bytes() for p in out.rglob("*")
             if p.is_file() and not p.name.endswith(".collecting") and "traces.collecting" not in p.parts}
    assert after == before  # the old manifest still points at its own traces


def test_collect_of_nothing_leaves_an_existing_run_alone(tmp_path: Path):
    cfg = config()
    _, result, _ = play(cells(cfg)[0], [accept_last] * 10)
    _fake_trial(tmp_path / "job", "t0", {"name": "claude-code", "model_name": "anthropic/x"}, result,
                "2026-11-27T00:00:00")
    out = tmp_path / "run"
    collect_jobs(cfg.model_dump(), config_hash(cfg), [tmp_path / "job"], out)
    manifest = (out / "manifest.jsonl").read_text()
    other = config(lineup=[cfg.lineup[0], {**cfg.lineup[1].model_dump(exclude_none=True), "model": "o/m"}])
    counts = collect_jobs(other.model_dump(), config_hash(other), [tmp_path / "job"], out)
    assert counts["kept"] == 0 and (out / "manifest.jsonl").read_text() == manifest


def test_collect_refuses_two_agents_with_the_same_focal_name(tmp_path: Path):
    cfg = config()
    for i, model in enumerate(["org/model", "org-model"]):
        _, result, _ = play(cells(cfg)[i], [accept_last] * 10)
        _fake_trial(tmp_path / "job", f"t{i}", {"name": "claude-code", "model_name": model}, result,
                    f"2026-11-27T00:0{i}:00")
    with pytest.raises(ValueError, match="both become the focal model"):
        collect_jobs(cfg.model_dump(), config_hash(cfg), [tmp_path / "job"], tmp_path / "run")


def _meeting_cells(**kw):
    cfg = load_config(ROOT / "configs" / "meeting-canary.yaml")
    chash = config_hash(cfg)
    return cfg, [replace(c, config_hash=chash)
                 for c in harbor_cells(build_grid(cfg), run_id=cfg.run_id, env=cfg.env,
                                       env_config=cfg.env_config, lineup=cfg.lineup, **kw)]


def test_external_arm_cells():
    _, hc = _meeting_cells(external_arm="harbor-user", external_replicates=2)
    ext = [c for c in hc if c.arm == "harbor-user"]
    assert len(ext) == 12 * 2 and {c.replicate for c in ext} == {0, 1}
    assert all(c.seats == {"contact": "external:harbor-user"} and c.lineup == [] for c in ext)
    with pytest.raises(ValueError, match="already exists"):
        _meeting_cells(external_arm="soc")


def test_external_arm_without_the_scripted_arm():
    cfg = load_config(ROOT / "configs" / "meeting-canary.yaml").model_copy(update={"run_arms": ["aoc"]})
    hc = harbor_cells(build_grid(cfg), run_id=cfg.run_id, env=cfg.env, env_config=cfg.env_config,
                      lineup=cfg.lineup, external_arm="harbor-user")
    ext = [c for c in hc if c.arm == "harbor-user"]
    assert {c.arm for c in hc} == {"aoc", "harbor-user"}
    assert sorted(c.seed for c in ext) == sorted({c.seed for c in hc if c.arm == "aoc"})
    assert all(c.lineup == [] and c.seats == {"contact": "external:harbor-user"} for c in ext)


def test_counterpart_played_from_outside_through_tools():
    """Both seats are played through tools, alternating, as Harbor's user
    agent (counterpart) and the agent under test would."""
    from parley_sim.envs import meeting as mt

    _, hc = _meeting_cells(external_arm="harbor-user")
    cell = next(c for c in hc if c.arm == "harbor-user")
    cp_token = "c" * 24
    with pytest.raises(ValueError, match="PARLEY_COUNTERPART_TOKEN"):
        HarborRuntime(RuntimeConfig(cell=cell, token=TOKEN))
    rt = HarborRuntime(RuntimeConfig(cell=cell, token=TOKEN, counterpart_token=cp_token))
    app = create_runtime_app(rt)

    async def go():
        async with app.router.lifespan_context(app):
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(transport=transport, base_url="http://parley-runtime:8000") as c:
                async def tool(name, **args):
                    r = await c.post("/mcp", headers=HEADERS, json={
                        "jsonrpc": "2.0", "id": 1, "method": "tools/call",
                        "params": {"name": name, "arguments": args}})
                    res = r.json()["result"]
                    return res.get("isError", False), res["content"][0]["text"]

                err, text = await tool("counterpart_start", token="wrong" * 5)
                assert err and "not for you" in text
                _, agent_view = await tool("start_episode")
                assert "assistant" in agent_view and "other side's turn" in agent_view
                assert agent_view.endswith(CHAT_NOTE)  # its chat replies reach the counterparty
                # Out of turn, the agent is told plainly that nothing was recorded.
                _, out = await tool("take_turn", action={"message": "Hi", "move": {"type": "none"}})
                assert out.startswith("Your action was NOT recorded") and out.endswith(CHAT_NOTE)
                _, cp_view = await tool("counterpart_start", token=cp_token)
                assert "counterpart_take_turn" in cp_view and CHAT_NOTE not in cp_view
                # A malformed counterpart action is a strike, and the counterpart is told at once.
                _, out = await tool("counterpart_take_turn", token=cp_token, action={"move": "x"})
                assert out.startswith("Your action could not be used") and CHAT_NOTE not in out
                _, out = await tool("take_turn", action={"message": "Your move.", "move": {"type": "none"}})
                inst = rt.episode.state.inst
                mutual = sorted(set(inst.free[mt.ASSISTANT]) & set(inst.free[mt.CONTACT]))
                _, out = await tool("counterpart_take_turn", token=cp_token, action={
                    "message": "Can we meet?", "move": {"type": "propose", "slot": mutual[0]}})
                assert "other side's turn" in out
                # Out of turn, the counterpart's move is not taken.
                _, out = await tool("counterpart_take_turn", token=cp_token, action={
                    "message": "again", "move": {"type": "none"}})
                assert out.startswith("Your action was NOT recorded")
                _, out = await tool("take_turn", action={"message": "Sure.", "move": {"type": "accept"}})
                assert "episode is over" in out
                forbidden = (await c.get("/result")).status_code
                return forbidden, (await c.get("/result", headers={"x-parley-token": TOKEN})).json()

    forbidden, result = asyncio.run(go())
    assert forbidden == 403
    assert result["completed"] == 1 and result["verified"] == 1 and result["no_deal"] == 0
    sources = [(l["seat"], l["raw"]["source"]) for l in result["trace"] if l["type"] == "decision"]
    assert sources == [("contact", "harbor-user"), ("assistant", "harbor")] * 2


def test_export_of_an_external_arm(tmp_path: Path):
    cfg, hc = _meeting_cells(external_arm="harbor-user")
    cell = next(c for c in hc if c.arm == "harbor-user")
    task = export_task(tmp_path, cell, chash=config_hash(cfg), wheel=_wheel(tmp_path))
    instruction = (task / "instruction.md").read_text()
    compose = (task / "environment" / "docker-compose.yaml").read_text()
    cp_token = instruction.split('token "')[1].split('"')[0]
    assert f'PARLEY_COUNTERPART_TOKEN: "{cp_token}"' in compose
    assert "counterpart_start" in instruction and "`contact` seat" in instruction
    assert "before prompting the assistant" in instruction
    assert '"propose"' in instruction  # the action format is inline
    # The result token stays out of the instruction (the counterparty must not read the score).
    toml = (task / "task.toml").read_text()
    result_token = toml.split('PARLEY_RESULT_TOKEN = "')[1].split('"')[0]
    assert result_token not in instruction


def _bridge_export(replies: list[str]) -> dict:
    """An ACP session export as Harbor's bridge writes it (abridged)."""
    messages = []
    for text in replies:
        messages.append({"User": {"content": [{"Text": "Your turn. Use your parley-runtime tools."}]}})
        messages.append({"Agent": {"content": [{"ToolUse": {"name": "take_turn", "raw_input": "{}"}},
                                               {"Text": text}]}})
    return {"session": {"state": {"messages": messages}}}


def test_bridge_replies_are_the_agents_text_only(tmp_path: Path):
    from parley_sim.adapters.harbor import bridge_replies

    path = tmp_path / "bridge-trajectory.json"
    assert bridge_replies(path) is None
    path.write_text(json.dumps(_bridge_export(["Try Fri 12:00 (9).", "Done, your turn."])))
    assert bridge_replies(path) == ["Try Fri 12:00 (9).", "Done, your turn."]


def test_collect_counts_chat_leaks_in_simulated_user_trials(tmp_path: Path):
    from parley_sim.envs import meeting as mt

    cfg, hc = _meeting_cells()
    cell = next(c for c in hc if c.arm == "soc")
    rt = HarborRuntime(RuntimeConfig(cell=cell, token=TOKEN))
    asyncio.run(rt.start())
    mine = rt.episode.state.inst.free[mt.ASSISTANT]
    best, value = max(mine.items(), key=lambda kv: kv[1])
    text = ""
    for _ in range(12):
        move = {"type": "propose", "slot": best}
        text = asyncio.run(rt.take_turn({"message": "How about it?", "move": move}))
        if "episode is over" in text:
            break
    result = rt.result()
    assert "episode is over" in text and ("Booked:" in text or "No meeting was booked" in text)
    _fake_trial(tmp_path / "job", "t0", {"name": "claude-code", "model_name": "anthropic/x"}, result,
                "2026-11-27T00:00:00")
    replies = [f"I proposed {best} (worth {value:g} to my manager).", "Done, your turn."]
    agent_dir = tmp_path / "job" / "t0" / "agent"
    agent_dir.mkdir()
    (agent_dir / "bridge-trajectory.json").write_text(json.dumps(_bridge_export(replies)))
    counts = collect_jobs(cfg.model_dump(), config_hash(cfg), [tmp_path / "job"], tmp_path / "run")
    assert counts["kept"] == 1 and counts["chat_replies"] == 2 and counts["chat_leaks"] == 1
    row = [json.loads(l) for l in (tmp_path / "run" / "manifest.jsonl").read_text().splitlines()][1]
    assert row["chat"] == {"replies": replies, "leaks": 1}


def test_an_environment_error_on_a_tool_move_ends_the_episode_cleanly():
    soc = next(c for c in cells(config()) if c.arm == "soc")
    rt = HarborRuntime(RuntimeConfig(cell=soc, token=TOKEN))
    asyncio.run(rt.start())

    def broken(*a, **kw):
        raise RuntimeError("environment bug")

    rt.episode.env.apply = broken
    move = {"message": "hi", "move": PROPOSE, "explanation": "x"}
    first = asyncio.run(rt.take_turn(move))
    again = asyncio.run(rt.take_turn(move))  # not a stale pending request
    assert "ended early" in first and "ended early" in again
    assert rt.pending is None and rt.result()["completed"] == 0


def test_chat_leaks_are_counted_for_unfinished_trials_too(tmp_path: Path):
    from parley_sim.envs import meeting as mt

    cfg, hc = _meeting_cells()
    cell = next(c for c in hc if c.arm == "soc")
    rt = HarborRuntime(RuntimeConfig(cell=cell, token=TOKEN))
    asyncio.run(rt.start())
    best, value = max(rt.episode.state.inst.free[mt.ASSISTANT].items(), key=lambda kv: kv[1])
    asyncio.run(rt.take_turn({"message": "Hi", "move": {"type": "propose", "slot": best}}))
    result = rt.result()  # the episode is not over
    assert result["completed"] == 0
    _fake_trial(tmp_path / "job", "t0", {"name": "claude-code", "model_name": "anthropic/x"}, result,
                "2026-11-27T00:00:00")
    (tmp_path / "job" / "t0" / "agent").mkdir()
    (tmp_path / "job" / "t0" / "agent" / "bridge-trajectory.json").write_text(
        json.dumps(_bridge_export([f"{best} is worth {value:g} to me."])))
    counts = collect_jobs(cfg.model_dump(), config_hash(cfg), [tmp_path / "job"], tmp_path / "run")
    assert counts["chat_replies"] == 1 and counts["chat_leaks"] == 1
