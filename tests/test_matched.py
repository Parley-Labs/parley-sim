import asyncio
import json

import pytest
from pydantic import ValidationError

from parley_sim.cli import smoke_config
from parley_sim.matched import MatchedConfig, build_grid, run_matched
from parley_sim.metrics import build_report, load_manifest


def cfg(**kw):
    base = smoke_config("0:4").model_dump()
    return MatchedConfig(**{**base, **kw})


def test_grid_pairs_twins_on_seed_and_focal_seat():
    cells = build_grid(cfg())
    assert len(cells) == 4 * (1 + 2)
    assert build_grid(cfg()) == cells
    for pair in {c.pair_key for c in cells}:
        group = [c for c in cells if c.pair_key == pair]
        assert {c.arm for c in group} == {"soc", "aoc"}
        assert len({(c.seed, c.focal_seat, c.seats[c.focal_seat]) for c in group}) == 1
    soc = [c.seats["customer"] for c in cells if c.arm == "soc"]
    assert set(soc) == {"policy:hardliner", "policy:reasonable", "policy:conceder"}


def test_config_rejects_bad_references():
    with pytest.raises(ValidationError):
        cfg(focal_models=["missing"])
    with pytest.raises(ValidationError):
        cfg(scripted_counterparts=["agent_scripted"])  # plays the wrong seat
    with pytest.raises(ValidationError):
        cfg(seeds="3:3")
    with pytest.raises(ValidationError):
        cfg(unknown_key=1)


def test_run_resume_and_config_guard(tmp_path):
    c = cfg()
    asyncio.run(run_matched(c, tmp_path))
    _, cells = load_manifest(tmp_path)
    assert len(cells) == 12 and all(r["status"] == "done" for r in cells)
    asyncio.run(run_matched(c.model_copy(update={"concurrency": 1}), tmp_path))
    assert load_manifest(tmp_path)[1] == cells  # nothing re-ran
    with pytest.raises(RuntimeError):
        asyncio.run(run_matched(cfg(seeds="0:5"), tmp_path))


class Walker:
    """Fake provider: always walks away, reporting 100 input tokens per call."""

    async def complete(self, *, system, user, schema):
        reply = {"message": "bye", "move": {"type": "walk_away"}, "explanation": "r"}
        return json.dumps(reply), {"usage": {"input_tokens": 100, "output_tokens": 0}}


def api_cfg(**kw):
    lineup = smoke_config("0:4").model_dump()["lineup"] + [
        {"name": "api-customer", "provider": "openai_compat", "model": "x"}]
    return cfg(lineup=lineup, adaptive_counterparts=["api-customer"], **kw)


def test_failed_cell_is_contained(tmp_path):
    def factory(spec):
        raise RuntimeError("no key")

    asyncio.run(run_matched(api_cfg(), tmp_path, client_factory=factory))
    _, cells = load_manifest(tmp_path)
    by_arm = {(r["arm"], r["status"]) for r in cells}
    assert by_arm == {("soc", "done"), ("aoc", "errored")}
    assert all("no key" in r["error"] for r in cells if r["status"] == "errored")


def test_spend_ceiling_skips_cells(tmp_path):
    asyncio.run(run_matched(api_cfg(concurrency=1, max_total_tokens=150), tmp_path,
                            client_factory=lambda spec: Walker()))
    _, cells = load_manifest(tmp_path)
    statuses = [r["status"] for r in cells]
    assert "skipped_budget" in statuses and "done" in statuses
    assert all(r["usage"]["input_tokens"] in (0, 100) for r in cells)


def test_report_shape(tmp_path):
    asyncio.run(run_matched(cfg(), tmp_path))
    report = build_report(tmp_path)
    m = report["comparisons"]["scripted-agent"]["aoc vs soc"]["deal"]
    assert report["arms"] == ["soc", "aoc"]
    assert m["n_pairs"] == 4 and m["kind"] == "relational"
    assert m["noise_floor"] == 0.0  # scripted stand-ins are deterministic
    assert set(report["acceptance_by_offer_quality"]["scripted-agent"]) == {"soc", "aoc"}


class Refuser:
    def __init__(self):
        self.calls = 0

    async def complete(self, *, system, user, schema):
        from parley_sim.drivers.models import ProviderRefusal
        self.calls += 1
        raise ProviderRefusal("reasoning_extraction")


class Suspended:
    def __init__(self):
        self.calls = 0

    async def complete(self, *, system, user, schema):
        self.calls += 1
        err = RuntimeError("model_access_suspended")
        err.status_code = 403
        raise err


@pytest.mark.parametrize("client_cls", [Refuser, Suspended])
def test_breaker_halts_the_run_and_limits_calls(tmp_path, client_cls):
    client = client_cls()
    asyncio.run(run_matched(api_cfg(concurrency=1, max_refusals=3), tmp_path,
                                       client_factory=lambda spec: client))
    _, cells = load_manifest(tmp_path)
    statuses = [r["status"] for r in cells]
    assert "skipped_halted" in statuses
    assert set(statuses) <= {"done", "errored", "skipped_halted"}
    assert statuses.count("skipped_halted") >= len(cells) - 3
    assert client.calls <= 3
    from parley_sim.cli import main
    assert main(["report", str(tmp_path)]) == 0


def test_cli_matched_exits_nonzero_when_halted(tmp_path, monkeypatch):
    import yaml

    from parley_sim import cli
    cfg_path = tmp_path / "c.yaml"
    cfg_path.write_text(yaml.safe_dump(api_cfg(concurrency=1).model_dump()))
    real = cli.run_matched
    monkeypatch.setattr(cli, "run_matched",
                        lambda c, out: real(c, out, client_factory=lambda s: Suspended()))
    assert cli.main(["matched", str(cfg_path), "--out", str(tmp_path / "run")]) == 2


def test_extra_arm_run_separately_joins_the_original_run(tmp_path):
    base = cfg()
    asyncio.run(run_matched(base, tmp_path / "pilot"))
    lineup = base.model_dump()["lineup"] + [
        {"name": "scripted-reasonable", "provider": "scripted", "model": "reasonable"}]
    extra = cfg(lineup=lineup, extra_arms={"persona": ["scripted-reasonable"]},
                run_arms=["persona"])
    asyncio.run(run_matched(extra, tmp_path / "extra"))
    _, cells = load_manifest(tmp_path / "extra")
    assert {c["arm"] for c in cells} == {"persona"} and len(cells) == 4 * 2
    report = build_report(tmp_path / "pilot", tmp_path / "extra")
    assert report["arms"] == ["soc", "persona", "aoc"]
    focal = report["comparisons"]["scripted-agent"]
    assert set(focal) == {"persona vs soc", "aoc vs soc", "aoc vs persona"}
    assert focal["aoc vs persona"]["deal"]["n_pairs"] == 4
    from parley_sim.metrics import render
    assert "aoc vs persona" in render(report)


def test_report_refuses_runs_on_different_scenarios(tmp_path):
    asyncio.run(run_matched(cfg(), tmp_path / "a"))
    asyncio.run(run_matched(cfg(env_config={"max_turns": 6}, extra_arms={"x": ["scripted-customer"]},
                                run_arms=["x"]), tmp_path / "b"))
    with pytest.raises(ValueError, match="different scenario"):
        build_report(tmp_path / "a", tmp_path / "b")


def test_config_rejects_bad_arms():
    with pytest.raises(ValidationError):
        cfg(extra_arms={"aoc": ["scripted-customer"]})
    with pytest.raises(ValidationError):
        cfg(run_arms=["nope"])
    lineup = cfg().model_dump()["lineup"]
    lineup[1]["persona"] = "usersim"
    with pytest.raises(ValidationError, match="persona_file"):
        cfg(lineup=lineup)


def test_equivalence_verdicts_and_diversity(tmp_path):
    from parley_sim.metrics import distinct2
    asyncio.run(run_matched(cfg(seeds="0:6"), tmp_path))
    report = build_report(tmp_path, margins={"deal": 1.0, "turns": 0.001})
    comp = report["comparisons"]["scripted-agent"]["aoc vs soc"]
    assert comp["deal"]["verdict"] in ("equivalent", "different")
    assert comp["deal"]["margin"] == 1.0 and len(comp["deal"]["delta_ci90"]) == 2
    assert comp["joint_efficiency"]["verdict"] in ("-", "different")
    div = report["diversity"]["scripted-agent"]
    assert set(div["per_arm"]) == {"soc", "aoc"}
    assert "opening_distinct2" in div["per_arm"]["soc"]
    assert 0 <= div["comparisons"]["aoc vs soc"]["opening_distinct2"]["p"] <= 1
    assert distinct2(["a b c", "a b c"]) == 0.5 and distinct2(["a b", "c d"]) == 1.0


def test_report_refuses_an_arm_defined_differently_in_two_runs(tmp_path):
    base = cfg()
    asyncio.run(run_matched(base, tmp_path / "a"))
    order = list(reversed(base.scripted_counterparts))
    asyncio.run(run_matched(cfg(run_id="b", seeds="4:8", scripted_counterparts=order,
                                run_arms=["soc"]), tmp_path / "b"))
    with pytest.raises(ValueError, match="defined differently"):
        build_report(tmp_path / "a", tmp_path / "b")


def test_halt_summary_counts_cells_not_the_halt_row(tmp_path):
    import json

    from parley_sim.cli import halt_summary
    m = tmp_path / "manifest.jsonl"
    rows = [{"type": "run"}, {"type": "cell", "status": "done"},
            {"type": "cell", "status": "skipped_halted", "error": "run halted: 3 refusals"},
            {"type": "halt", "reason": "3 refusals"}]
    m.write_text("".join(json.dumps(r) + "\n" for r in rows))
    reason, halted = halt_summary(m)
    assert reason == "3 refusals" and len(halted) == 1
    m.write_text(json.dumps({"type": "run"}) + "\n" + json.dumps({"type": "halt", "reason": "403"}) + "\n")
    assert halt_summary(m) == ("403", [])
    m.write_text(json.dumps({"type": "run"}) + "\n")
    assert halt_summary(m) is None


def test_report_refuses_a_focal_name_that_means_different_models(tmp_path):
    base = cfg()
    asyncio.run(run_matched(base, tmp_path / "a"))
    lineup = base.model_dump()["lineup"]
    lineup[0] = {**lineup[0], "max_tokens": 1234}  # same focal name, different settings
    lineup.append({"name": "scripted-reasonable", "provider": "scripted", "model": "reasonable"})
    other = cfg(lineup=lineup, extra_arms={"persona": ["scripted-reasonable"]},
                run_arms=["persona"])
    asyncio.run(run_matched(other, tmp_path / "b"))
    with pytest.raises(ValueError, match="configured differently"):
        build_report(tmp_path / "a", tmp_path / "b")


def test_old_manifests_and_explicit_defaults_join_but_real_differences_do_not(tmp_path):
    import json as _json
    base = cfg()
    asyncio.run(run_matched(base, tmp_path / "a"))
    lineup = base.model_dump()["lineup"] + [
        {"name": "scripted-reasonable", "provider": "scripted", "model": "reasonable"}]
    asyncio.run(run_matched(cfg(lineup=lineup, extra_arms={"persona": ["scripted-reasonable"]},
                                run_arms=["persona"]), tmp_path / "b"))
    # Make run "a" look like a manifest written before `conversation` existed.
    m = tmp_path / "a" / "manifest.jsonl"
    rows = [_json.loads(l) for l in m.read_text().splitlines()]
    for spec in rows[0]["config"]["lineup"]:
        spec.pop("conversation", None)
    m.write_text("".join(_json.dumps(r) + "\n" for r in rows))
    build_report(tmp_path / "a", tmp_path / "b")  # missing and explicit default are the same
    asyncio.run(run_matched(base, tmp_path / "a"))  # resuming the old run keeps it joinable
    build_report(tmp_path / "a", tmp_path / "b")
    lineup[0] = {**lineup[0], "conversation": True}
    asyncio.run(run_matched(cfg(run_id="c", lineup=lineup, extra_arms={"persona": ["scripted-reasonable"]},
                                run_arms=["persona"]), tmp_path / "c"))
    with pytest.raises(ValueError, match="configured differently"):
        build_report(tmp_path / "a", tmp_path / "c")


def test_report_refuses_a_persona_arm_built_from_different_panels(tmp_path):
    import json as _json
    lineup = cfg().model_dump()["lineup"] + [
        {"name": "scripted-reasonable", "provider": "scripted", "model": "reasonable"}]
    for name, seeds in (("b", "0:2"), ("c", "2:4")):
        asyncio.run(run_matched(cfg(run_id=name, seeds=seeds, lineup=lineup,
                                    extra_arms={"persona": ["scripted-reasonable"]},
                                    run_arms=["persona"]), tmp_path / name))
    # Mark the arm as persona-driven, with a different panel in each run.
    for name in ("b", "c"):
        m = tmp_path / name / "manifest.jsonl"
        rows = [_json.loads(l) for l in m.read_text().splitlines()]
        spec = next(s for s in rows[0]["config"]["lineup"] if s["name"] == "scripted-reasonable")
        spec["persona"] = "usersim"
        rows[0]["persona_digest"] = f"panel-{name}"
        m.write_text("".join(_json.dumps(r) + "\n" for r in rows))
    with pytest.raises(ValueError, match="different persona panel"):
        build_report(tmp_path / "b", tmp_path / "c")
