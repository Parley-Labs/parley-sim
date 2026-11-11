import json
from pathlib import Path

import pytest

from parley_sim.envs.support_desk import SupportDesk, make_policy
from parley_sim.runtime import DecisionRequest, DecisionResult, Episode, EpisodeComplete, EpisodeSpec, verify
from parley_sim.trace import JsonlWriter, is_complete, read_trace

GOLDEN = Path(__file__).parent / "golden" / "support_desk_seed7_reasonable.jsonl"


def scripted_episode(seed=7, persona="reasonable", sink=None):
    spec = EpisodeSpec(seed=seed, seats={"agent": "policy:agent_scripted",
                                         "customer": f"policy:{persona}"})
    policies = {"agent": make_policy("agent_scripted", "agent"),
                "customer": make_policy(persona, "customer")}
    ep = Episode(SupportDesk(), spec, policies, sink=sink)
    return ep, ep.advance()


def test_scripted_episode_runs_to_completion_and_replays():
    ep, done = scripted_episode()
    assert isinstance(done, EpisodeComplete) and done.status == "complete"
    assert [l["seq"] for l in ep.lines] == list(range(len(ep.lines)))
    assert ep.lines[0]["type"] == "header" and ep.lines[-1]["type"] == "footer"
    assert verify(SupportDesk(), ep.lines) == []


def test_trace_matches_golden(tmp_path):
    with JsonlWriter(tmp_path / "t.jsonl") as w:
        scripted_episode(sink=w)
    produced = read_trace(tmp_path / "t.jsonl")
    if not GOLDEN.exists():  # first run materializes the fixture; commit it
        GOLDEN.parent.mkdir(exist_ok=True)
        GOLDEN.write_text("".join(json.dumps(l, sort_keys=True) + "\n" for l in produced))
    assert produced == read_trace(GOLDEN)
    assert is_complete(tmp_path / "t.jsonl")


def test_external_seat_inverts_control():
    spec = EpisodeSpec(seed=3, seats={"agent": "model:test", "customer": "policy:conceder"},
                       focal_seat="agent")
    ep = Episode(SupportDesk(), spec, {"customer": make_policy("conceder", "customer")})
    event = ep.advance()
    n_requests = 0
    while not isinstance(event, EpisodeComplete):
        assert isinstance(event, DecisionRequest) and event.seat == "agent"
        n_requests += 1
        move = ({"type": "accept"} if event.observation["their_standing_offer"]
                else {"type": "none"})
        event = ep.advance(DecisionResult(event.request_id, {"message": "ok", "move": move},
                                          reasoning="r"))
    assert n_requests == 1
    assert ep.lines[-1]["final_state"]["end_reason"] == "accepted"
    decisions = [l for l in ep.lines if l["type"] == "decision"]
    assert [d["source"] for d in decisions] == ["scripted", "external"]
    assert decisions[1]["context"]["standing_offer"] is not None
    assert verify(SupportDesk(), ep.lines) == []


def test_illegal_and_missing_actions_fall_back_and_are_recorded():
    spec = EpisodeSpec(seed=3, seats={"agent": "model:test", "customer": "policy:hardliner"})
    ep = Episode(SupportDesk(), spec, {"customer": make_policy("hardliner", "customer")})
    req = ep.advance()
    over = {"discount_pct": 15, "credit_usd": 0, "upgrade": "none", "term": "monthly"}
    req = ep.advance(DecisionResult(req.request_id,
                                    {"message": "", "move": {"type": "propose", "deal": over}}))
    req = ep.advance(DecisionResult(req.request_id, None, strike="timeout"))
    decisions = [l for l in ep.lines if l["type"] == "decision" and l["source"] == "external"]
    assert decisions[0]["strike"] == "illegal_action"
    assert decisions[0]["applied"]["move"] == {"type": "none"}
    assert decisions[0]["submitted"]["move"]["type"] == "propose"
    assert decisions[1]["strike"] == "timeout"


def test_result_must_answer_the_pending_request():
    spec = EpisodeSpec(seed=3, seats={"agent": "model:test", "customer": "policy:hardliner"})
    ep = Episode(SupportDesk(), spec, {"customer": make_policy("hardliner", "customer")})
    req = ep.advance()
    with pytest.raises(ValueError):
        ep.advance(DecisionResult("wrong", None, strike="timeout"))
    with pytest.raises(ValueError):
        ep.advance()
    assert isinstance(req, DecisionRequest)


def test_spec_must_cover_env_seats_and_policies():
    with pytest.raises(ValueError):
        Episode(SupportDesk(), EpisodeSpec(seed=0, seats={"agent": "model:x"}), {})
    with pytest.raises(ValueError):
        Episode(SupportDesk(), EpisodeSpec(seed=0, seats={"agent": "model:x",
                                                          "customer": "policy:conceder"}), {})


def test_abort_writes_errored_footer():
    spec = EpisodeSpec(seed=3, seats={"agent": "model:test", "customer": "policy:hardliner"})
    ep = Episode(SupportDesk(), spec, {"customer": make_policy("hardliner", "customer")})
    ep.advance()
    done = ep.abort("boom")
    assert done.status == "errored" and ep.lines[-1]["error"] == "boom"
