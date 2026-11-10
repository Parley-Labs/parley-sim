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
        reply = {"reasoning": "r", "message": "bye", "move": {"type": "walk_away"}}
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
    m = report["metrics"]["scripted-agent"]["deal"]
    assert m["n_pairs"] == 4 and m["kind"] == "relational"
    assert m["noise_floor"] == 0.0  # scripted stand-ins are deterministic
    assert set(report["acceptance_by_offer_quality"]["scripted-agent"]) == {"soc", "aoc"}
