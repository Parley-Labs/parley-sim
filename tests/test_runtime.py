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


@pytest.mark.parametrize("edit,problem", [
    (lambda ls: ls[-1]["final_state"].__setitem__("turn", 999), "final state differs"),
    (lambda ls: ls[0].__setitem__("initial_state_hash", "x"), "initial state hash"),
    (lambda ls: ls[-1].__setitem__("n_strikes", 5), "n_strikes"),
    (lambda ls: ls[-1].__setitem__("n_decisions", 1), "n_decisions"),
])
def test_verify_catches_edits_to_recorded_state_and_counts(tmp_path, edit, problem):
    import asyncio

    from parley_sim.cli import smoke_config
    from parley_sim.envs.support_desk import SupportDesk
    from parley_sim.matched import run_matched
    from parley_sim.runtime import verify
    from parley_sim.trace import read_trace
    asyncio.run(run_matched(smoke_config("0:1"), tmp_path))
    lines = read_trace(sorted((tmp_path / "traces").glob("*.jsonl"))[0])
    assert verify(SupportDesk(), lines) == []
    edit(lines)
    assert any(problem in p for p in verify(SupportDesk(), lines))


def _smoke_soc_trace(tmp_path):
    """A complete trace with a scripted seat (the customer policy) and an
    external one (the agent: a scripted lineup entry played through the model
    path)."""
    import asyncio

    from parley_sim.cli import smoke_config
    from parley_sim.matched import run_matched
    asyncio.run(run_matched(smoke_config("0:1"), tmp_path))
    return read_trace(next((tmp_path / "traces").glob("*-soc-r0.jsonl")))


def _first(lines, kind, **match):
    return next(l for l in lines if l["type"] == kind
                and all(l.get(k) == v for k, v in match.items()))


def _cut_after_first_decision(lines):
    """Keep the header, the first decision and its events, and close with a
    footer consistent with that state that still says `complete`."""
    from parley_sim.core.hashing import digest
    env = SupportDesk()
    spec = lines[0]["spec"]
    state = env.initial_state(env.generate(spec["seed"], spec["config"]))
    first = _first(lines, "decision")
    state, events = env.apply(state, first["seat"], first["applied"])
    kept = lines[:2 + len(events)]
    kept.append({"seq": len(kept), "type": "footer", "status": "complete",
                 "outcomes": env.outcome(state), "n_decisions": 1,
                 "n_strikes": int(first["strike"] is not None),
                 "final_state": env.state_to_json(state),
                 "final_state_hash": digest(env.state_to_json(state))})
    lines[:] = kept


def _rename_customer_policy(lines):
    """Rewrite the scripted customer's policy name consistently everywhere."""
    new = "conceder" if lines[0]["policies"]["customer"]["name"] != "conceder" else "hardliner"
    lines[0]["policies"]["customer"]["name"] = new
    lines[0]["spec"]["seats"]["customer"] = f"policy:{new}"
    for l in lines:
        if l["type"] == "decision" and l["seat"] == "customer":
            l["label"] = f"policy:{new}"


def _drop_an_event(lines):
    del lines[next(i for i, l in enumerate(lines) if l["type"] == "event")]
    for j, l in enumerate(lines):
        l["seq"] = j


@pytest.mark.parametrize("edit,problem", [
    # B1: logged decision fields
    (lambda ls: _first(ls, "decision", seat="agent")["observation"]["private"]
     .__setitem__("value_if_no_deal", 1), "observation differs"),
    (lambda ls: _first(ls, "decision", seat="agent")["context"]
     .__setitem__("walkaway_value", 1), "decision context differs"),
    (lambda ls: _first(ls, "decision", seat="agent").__setitem__("turn", 5), "turn 5"),
    (lambda ls: _first(ls, "decision", seat="agent").__setitem__("label", "model:other"), "label"),
    (lambda ls: _first(ls, "decision", seat="customer").__setitem__("source", "external"),
     "source 'external'"),
    (lambda ls: _first(ls, "decision", seat="agent").__setitem__("seat", "customer"),
     "to act"),
    # B2: event payloads
    (lambda ls: _first(ls, "event", event_type="move").__setitem__("message", "edited"),
     "event differs"),
    (lambda ls: _first(ls, "event", event_type="proposal")["deal"].__setitem__("credit_usd", 999),
     "event differs"),
    # B3: an event deleted
    (_drop_an_event, "missing"),
    # B4: game cut after one decision, footer still says complete
    (_cut_after_first_decision, "still to act"),
    # B5: header seats, focal seat and policies
    (lambda ls: ls[0]["spec"]["seats"].__setitem__("ghost", "model:x"), "header seats"),
    (lambda ls: ls[0]["spec"].__setitem__("focal_seat", "ghost"), "focal seat"),
    (lambda ls: ls[0]["policies"]["customer"].__setitem__("name", "conceder"), "its policy is"),
    (lambda ls: ls[0]["policies"]["customer"]["params"].__setitem__("beta", 9.0),
     "differs from header"),
    (_rename_customer_policy, "makes a different move"),
    (lambda ls: ls[0]["policies"].pop("customer"), "no policy for it"),
])
def test_verify_catches_tampering(tmp_path, edit, problem):
    from parley_sim.envs.support_desk import make_policy as policies
    lines = _smoke_soc_trace(tmp_path)
    assert verify(SupportDesk(), lines, make_policy=policies) == []
    edit(lines)
    problems = verify(SupportDesk(), lines, make_policy=policies)
    assert any(problem in p for p in problems), problems


def test_verify_compares_stored_json_not_python_types(monkeypatch):
    """An env whose instance and ground truth hold tuples still verifies
    against the JSON-loaded header."""
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "examples"))
    import haggle

    class TupleHaggle(haggle.Haggle):
        def instance_to_json(self, inst):
            return {**super().instance_to_json(inst), "range": (inst.cost, inst.value)}

        def ground_truth(self, inst):
            return {**super().ground_truth(inst), "zopa": (inst.cost, inst.value)}

    env = TupleHaggle()
    spec = EpisodeSpec(seed=1, seats={"seller": "policy:flexible", "buyer": "policy:firm"},
                       focal_seat="seller")
    ep = Episode(env, spec, {"seller": haggle.make_policy("flexible", "seller"),
                             "buyer": haggle.make_policy("firm", "buyer")})
    ep.advance()
    stored = [json.loads(json.dumps(l)) for l in ep.lines]
    assert verify(env, stored, make_policy=haggle.make_policy) == []


def test_lone_surrogate_reply_is_a_malformed_strike(tmp_path):
    spec = EpisodeSpec(seed=3, seats={"agent": "model:test", "customer": "policy:hardliner"})
    path = tmp_path / "t.jsonl"
    with JsonlWriter(path) as w:
        ep = Episode(SupportDesk(), spec, {"customer": make_policy("hardliner", "customer")},
                     sink=w)
        req = ep.advance()
        bad = json.loads('{"message": "Thanks \\ud83d", "move": {"type": "none"}}')
        event = ep.advance(DecisionResult(req.request_id, bad, explanation="\ud83d",
                                          raw={"payload": "Thanks \ud83d"}))
        while not isinstance(event, EpisodeComplete):
            event = ep.advance(DecisionResult(event.request_id,
                                              {"message": "", "move": {"type": "walk_away"}}))
    lines = read_trace(path)
    assert is_complete(path) and lines[-1]["status"] == "complete"
    d = _first(lines, "decision", seat="agent")
    assert d["strike"] == "malformed_output"
    assert d["applied"] == {"message": "", "move": {"type": "none"}}
    assert d["submitted"]["message"] == "Thanks \\ud83d"  # kept, escaped
    assert verify(SupportDesk(), lines) == []


class _Unserializable(SupportDesk):
    def state_to_json(self, state):
        if state.turn > 0:
            raise TypeError("cannot serialize")
        return super().state_to_json(state)


def test_errored_footer_is_written_when_the_state_cannot_be_serialized(tmp_path):
    spec = EpisodeSpec(seed=3, seats={"agent": "model:test", "customer": "policy:hardliner"})
    path = tmp_path / "t.jsonl"
    with JsonlWriter(path) as w:
        ep = Episode(_Unserializable(), spec, {"customer": make_policy("hardliner", "customer")},
                     sink=w)
        with pytest.raises(TypeError):
            ep.advance()  # the state after the customer's opening move cannot be hashed
        done = ep.abort("boom")
    footer = read_trace(path)[-1]
    assert done.status == "errored" and footer["type"] == "footer"
    assert footer["final_state"] is None and "cannot serialize" in footer["error"]


@pytest.mark.parametrize("content,problem", [
    ("", "empty"),
    ('{"seq": 0, "type": "header"', "truncated or unparsable"),
    ('{"seq": 0, "type": "header", "env": {"name": "support_desk"}}\n', "no footer"),
])
def test_cli_verify_reports_bad_files_without_a_traceback(tmp_path, capsys, content, problem):
    from parley_sim.cli import main
    path = tmp_path / "t.jsonl"
    path.write_text(content)
    assert main(["verify", str(path)]) == 1
    assert problem in capsys.readouterr().out


def test_verify_reports_a_deeply_nested_trace_instead_of_crashing(tmp_path):
    from parley_sim.cli import verify_file
    from parley_sim.trace import is_complete
    bad = tmp_path / "deep.jsonl"
    bad.write_text("[" * 200_000 + "]" * 200_000 + "\n")
    problems = verify_file(bad)
    assert problems and not is_complete(bad)


def test_secret_seed_is_keyed_deterministic_and_63_bit():
    import pytest

    from parley_sim.core.rng import secret_seed
    a = secret_seed("s3cret", 1000)
    assert a == secret_seed("s3cret", 1000) and 0 <= a < 2**63
    assert a != secret_seed("other", 1000) and a != secret_seed("s3cret", 1001) and a != 1000
    with pytest.raises(ValueError):
        secret_seed("", 1000)
