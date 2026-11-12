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


def _persona_runs(tmp_path, panels, *, focal=False):
    """Two runs on disjoint seeds joined into one report, with the focal model
    (or the "persona" arm) marked persona-conditioned and the given panel
    digest recorded per run (None: not recorded)."""
    import json as _json
    lineup = cfg().model_dump()["lineup"] + [
        {"name": "scripted-reasonable", "provider": "scripted", "model": "reasonable"}]
    focal_name = cfg().focal_models[0]
    marked = focal_name if focal else "scripted-reasonable"
    for (name, seeds), panel in zip((("b", "0:2"), ("c", "2:4")), panels, strict=True):
        asyncio.run(run_matched(cfg(run_id=name, seeds=seeds, lineup=lineup,
                                    extra_arms={"persona": ["scripted-reasonable"]},
                                    run_arms=["persona"]), tmp_path / name))
        m = tmp_path / name / "manifest.jsonl"
        rows = [_json.loads(l) for l in m.read_text().splitlines()]
        next(s for s in rows[0]["config"]["lineup"] if s["name"] == marked)["persona"] = "usersim"
        if panel is not None:
            rows[0]["persona_digest"] = panel
        m.write_text("".join(_json.dumps(r) + "\n" for r in rows))
    return tmp_path / "b", tmp_path / "c"


def test_report_refuses_a_persona_arm_built_from_different_panels(tmp_path):
    with pytest.raises(ValueError, match="arm 'persona' uses a different persona panel"):
        build_report(*_persona_runs(tmp_path, ["panel-b", "panel-c"]))


def test_report_refuses_a_persona_focal_model_built_from_different_panels(tmp_path):
    with pytest.raises(ValueError, match=r"focal model .* uses a different persona panel"):
        build_report(*_persona_runs(tmp_path, ["panel-b", "panel-c"], focal=True))


def test_report_refuses_a_persona_join_without_recorded_panels(tmp_path):
    with pytest.raises(ValueError, match="does not record its persona panel"):
        build_report(*_persona_runs(tmp_path, ["panel-b", None]))


def test_report_joins_persona_runs_on_the_same_panel(tmp_path):
    report = build_report(*_persona_runs(tmp_path, ["panel-x", "panel-x"]))
    assert "persona" in report["arms"]


def all_api_cfg(**kw):
    """Every model seat needs a provider client: no cell can finish offline."""
    api = {"provider": "openai_compat", "base_url": "https://gateway.invalid/v1",
           "api_key_env": "PARLEY_TEST_UNSET_KEY"}
    lineup = [{"name": "api-agent", "model": "x", **api},
              {"name": "api-customer", "model": "y", **api}]
    return cfg(lineup=lineup, focal_models=["api-agent"],
               adaptive_counterparts=["api-customer"], **kw)


def _cli_matched(tmp_path, monkeypatch, config, factory=None):
    import yaml

    from parley_sim import cli
    cfg_path = tmp_path / "c.yaml"
    cfg_path.write_text(yaml.safe_dump(config.model_dump()))
    if factory is not None:
        real = cli.run_matched
        monkeypatch.setattr(cli, "run_matched",
                            lambda c, out: real(c, out, client_factory=factory))
    return cli.main(["matched", str(cfg_path), "--out", str(tmp_path / "run")])


class Unauthorized(Exception):
    status_code = 401


def test_cli_matched_halts_cleanly_when_every_client_is_refused(tmp_path, monkeypatch, capsys):
    def factory(spec):
        raise Unauthorized("invalid api key")

    code = _cli_matched(tmp_path, monkeypatch, all_api_cfg(concurrency=1), factory)
    out = capsys.readouterr().out
    assert code == 2
    assert out.index("RUN HALTED") < out.index("# smoke")  # the summary comes first
    assert "invalid api key" in out and "No completed episodes" in out
    _, cells = load_manifest(tmp_path / "run")
    assert {r["status"] for r in cells} == {"errored", "skipped_halted"}


def test_cli_matched_with_a_missing_key_exits_1_and_says_why(tmp_path, monkeypatch, capsys):
    monkeypatch.delenv("PARLEY_TEST_UNSET_KEY", raising=False)
    assert _cli_matched(tmp_path, monkeypatch, all_api_cfg()) == 1  # the real client factory
    out = capsys.readouterr().out
    assert "12 cells errored" in out and "PARLEY_TEST_UNSET_KEY is not set" in out
    assert "No completed episodes" in out


def test_cli_matched_exits_1_when_some_cells_errored(tmp_path, monkeypatch, capsys):
    def factory(spec):
        raise RuntimeError("no key")

    assert _cli_matched(tmp_path, monkeypatch, api_cfg(), factory) == 1
    out = capsys.readouterr().out
    assert "8 x RuntimeError('no key')" in out and "Cells: done=4, errored=8" in out


def test_config_rejects_the_harness_provider():
    lineup = smoke_config("0:4").model_dump()["lineup"] + [
        {"name": "gym-agent", "provider": "harness", "model": "policy"}]
    with pytest.raises(ValidationError, match="harness"):
        cfg(lineup=lineup)


class OneCallThenAbort:
    """Answers once (100 input tokens), then fails the episode, so each cell
    ends errored with usage recorded in its trace."""

    def __init__(self):
        self.calls = 0

    async def complete(self, *, system, user, schema, **kw):
        from parley_sim.drivers.models import EpisodeAbort
        self.calls += 1
        if self.calls > 1:
            raise EpisodeAbort("upstream gone", retryable=True)
        reply = {"message": "hm", "move": {"type": "none"}, "explanation": "r"}
        return json.dumps(reply), {"usage": {"input_tokens": 100, "output_tokens": 0}}


def test_token_cap_counts_usage_across_resumes(tmp_path):
    c = api_cfg(concurrency=1)
    for _ in range(2):  # every aoc cell (8) errors after spending 100 tokens
        asyncio.run(run_matched(c, tmp_path, client_factory=lambda s: OneCallThenAbort()))
    run, cells = load_manifest(tmp_path)
    assert run["prior_usage"]["input_tokens"] == 800  # the first invocation's errored cells
    capped = c.model_copy(update={"max_total_tokens": 2000})  # operational: same config hash
    asyncio.run(run_matched(capped, tmp_path, client_factory=lambda s: OneCallThenAbort()))
    run, cells = load_manifest(tmp_path)
    assert run["prior_usage"]["input_tokens"] == 1600
    statuses = [r["status"] for r in cells if r["arm"] == "aoc"]
    assert statuses.count("errored") == 4 and statuses.count("skipped_budget") == 4
    report = build_report(tmp_path)
    assert report["spend"]["input_tokens"] == 2000  # dropped attempts included


def test_torn_last_manifest_line_is_skipped_then_healed(tmp_path, capsys):
    asyncio.run(run_matched(cfg(), tmp_path))
    manifest = tmp_path / "manifest.jsonl"
    with manifest.open("a") as f:
        f.write('{"type": "cell", "cell_key": "cut')
    assert build_report(tmp_path)["cells"] == {"done": 12}
    assert "torn last line" in capsys.readouterr().err
    asyncio.run(run_matched(cfg(), tmp_path))  # resume
    assert all(l.strip().endswith("}") for l in manifest.read_text().splitlines())
    lines = manifest.read_text().splitlines()
    manifest.write_text("\n".join([*lines[:2], "{not json", *lines[2:]]) + "\n")
    with pytest.raises(json.JSONDecodeError):  # only the last line may be torn
        load_manifest(tmp_path)


def test_resume_removes_traces_of_cells_it_does_not_run(tmp_path):
    c = api_cfg(concurrency=1)
    asyncio.run(run_matched(c, tmp_path, client_factory=lambda s: OneCallThenAbort()))
    traces = tmp_path / "traces"
    assert len(list(traces.glob("*-aoc-*.jsonl"))) == 8  # errored, with traces
    asyncio.run(run_matched(c.model_copy(update={"max_total_tokens": 1}), tmp_path,
                            client_factory=lambda s: OneCallThenAbort()))
    _, cells = load_manifest(tmp_path)
    assert {r["status"] for r in cells if r["arm"] == "aoc"} == {"skipped_budget"}
    assert not list(traces.glob("*-aoc-*.jsonl"))
    assert len(list(traces.glob("*-soc-*.jsonl"))) == 4


def test_a_single_pair_never_clears(tmp_path):
    asyncio.run(run_matched(cfg(seeds="1:2"), tmp_path))
    report = build_report(tmp_path, margins={"turns": 0.5})
    comp = report["comparisons"]["scripted-agent"]["aoc vs soc"]
    assert comp["turns"]["delta_mean"] != 0  # a real difference, but one pair
    assert all(not m["clears"] for m in comp.values())
    assert {m["verdict"] for m in comp.values()} == {"-", "inconclusive"}
    assert comp["turns"]["verdict"] == "inconclusive"


def test_bootstrap_interval_uses_the_intended_order_statistics():
    import random
    import statistics

    from parley_sim.metrics import bootstrap_ci
    values = [i ** 1.5 for i in range(30)]
    rng = random.Random(0)
    means = sorted(statistics.fmean(rng.choices(values, k=30)) for _ in range(2000))
    assert means[99] != means[100]  # int((1 - 0.90) / 2 * 2000) would pick 99
    assert bootstrap_ci(values, level=0.90) == (means[100], means[1899])
    assert bootstrap_ci(values) == (means[50], means[1949])


def test_unknown_margin_metric_is_an_error(tmp_path, capsys):
    from parley_sim.cli import main
    asyncio.run(run_matched(cfg(), tmp_path))
    with pytest.raises(ValueError, match="focal_surplus_shar"):
        build_report(tmp_path, margins={"focal_surplus_shar": 0.08})
    assert main(["report", str(tmp_path), "--margins", "focal_surplus_shar=0.08"]) == 2
    assert "unknown metric" in capsys.readouterr().err
    assert main(["report", str(tmp_path), "--margins", "turns"]) == 2


def test_report_env_flag_must_match_the_run(tmp_path, capsys):
    from parley_sim.cli import main
    asyncio.run(run_matched(cfg(), tmp_path))
    assert main(["report", str(tmp_path), "--env", "support_desk"]) == 0
    assert main(["report", str(tmp_path), "--env", "test_env_plugin:ENTRY"]) == 2
    assert "was run on 'support_desk'" in capsys.readouterr().err


def two_member_cfg(**kw):
    """Two-member counterpart arms whose members correspond: aoc has the
    plain customers, persona the same policies with a different voice."""
    lineup = smoke_config("0:4").model_dump()["lineup"] + [
        {"name": "plain-a", "provider": "scripted", "model": "conceder"},
        {"name": "plain-b", "provider": "scripted", "model": "hardliner"},
        {"name": "persona-a", "provider": "scripted", "model": "conceder"},
        {"name": "persona-b", "provider": "scripted", "model": "hardliner"}]
    return cfg(lineup=lineup, adaptive_counterparts=["plain-a", "plain-b"],
               extra_arms={"persona": ["persona-a", "persona-b"]}, **kw)


@pytest.mark.parametrize("persona_seeds,ok", [
    ("0:4", True),    # same seed list
    ("2:4", True),    # starts two seeds later: same point of a two-member rotation
    ("1:4", False),   # one seed later: plain-a would meet persona-b on every seed
])
def test_report_refuses_joins_that_misalign_counterpart_rotations(tmp_path, persona_seeds, ok):
    asyncio.run(run_matched(two_member_cfg(run_arms=["soc", "aoc"]), tmp_path / "a"))
    asyncio.run(run_matched(two_member_cfg(run_id="b", seeds=persona_seeds,
                                           run_arms=["persona"]), tmp_path / "b"))
    if ok:
        report = build_report(tmp_path / "a", tmp_path / "b")
        assert report["arms"] == ["soc", "persona", "aoc"]
    else:
        with pytest.raises(ValueError, match="pairing no single run makes"):
            build_report(tmp_path / "a", tmp_path / "b")


def test_report_joins_offset_runs_where_arms_share_no_rotation(tmp_path):
    """Three scripted counterparts against a two-member arm (gcd 1): a single
    run meets every combination, so an offset join is not a confound."""
    asyncio.run(run_matched(two_member_cfg(run_arms=["soc"]), tmp_path / "a"))
    asyncio.run(run_matched(two_member_cfg(run_id="b", seeds="1:4", run_arms=["aoc"]),
                            tmp_path / "b"))
    assert build_report(tmp_path / "a", tmp_path / "b")["arms"] == ["soc", "aoc"]


def test_render_shows_opening_counts_decision_counts_and_small_p(tmp_path):
    from parley_sim.metrics import render
    asyncio.run(run_matched(cfg(seeds="0:6"), tmp_path))
    report = build_report(tmp_path)
    div = report["diversity"]["scripted-agent"]
    assert div["per_arm"]["soc"]["n_openings"] == 6
    report["diversity"]["scripted-agent"]["comparisons"]["aoc vs soc"]["opening_distinct2"]["p"] = 1 / 2001
    text = render(report)
    assert "| n_openings |" in text
    assert "(p<0.001, n=6)" in text and "p=0.000" not in text
    assert "| decisions facing an offer |" in text and "| offers |" not in text
