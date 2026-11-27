"""The meeting_scheduling environment."""

import asyncio
import json
from pathlib import Path

import pytest

from parley_sim import envs
from parley_sim.core.protocol import IllegalAction, MalformedOutput
from parley_sim.envs import meeting as mt
from parley_sim.matched import load_config, run_matched
from parley_sim.metrics import build_report
from parley_sim.runtime.episode import verify
from parley_sim.trace import read_trace

ROOT = Path(__file__).resolve().parents[1]


def test_scenarios_are_seeded_and_feasible():
    env = mt.MeetingScheduling()
    for seed in range(30):
        a, b = env.generate(seed, {}), env.generate(seed, {})
        assert env.instance_to_json(a) == env.instance_to_json(b)
        gt = env.ground_truth(a)
        assert len(gt["mutual_slots"]) >= 4
        assert all(1 <= v <= 10 for side in a.free.values() for v in side.values())
    assert env.instance_to_json(env.generate(1, {})) != env.instance_to_json(env.generate(2, {}))


def _state(seed=3):
    env = mt.MeetingScheduling()
    return env, env.initial_state(env.generate(seed, {}))


def test_rules_are_enforced():
    env, s = _state()
    inst = s.inst
    busy_c = next(x for x in mt.SLOTS if x not in inst.free[mt.CONTACT])
    with pytest.raises(IllegalAction, match="busy"):
        env.apply(s, mt.CONTACT, {"message": "", "move": {"type": "propose", "slot": busy_c}})
    with pytest.raises(IllegalAction, match="not assistant's turn"):
        env.apply(s, mt.ASSISTANT, {"message": "", "move": {"type": "none"}})
    with pytest.raises(IllegalAction, match="no proposal"):
        env.apply(s, mt.CONTACT, {"message": "", "move": {"type": "accept"}})
    with pytest.raises(IllegalAction, match="slot must be"):
        env.apply(s, mt.CONTACT, {"message": "", "move": {"type": "propose", "slot": "Sat 10:00"}})
    # The contact proposes a slot the assistant is busy for: the assistant can't accept it.
    clash = next(x for x in inst.free[mt.CONTACT] if x not in inst.free[mt.ASSISTANT])
    s2, events = env.apply(s, mt.CONTACT, {"message": "hi", "move": {"type": "propose", "slot": clash}})
    assert events[0]["move"] == {"type": "propose", "slot": clash}
    assert env.observe(s2, mt.ASSISTANT)["their_proposal_value"] is None
    with pytest.raises(IllegalAction, match="busy"):
        env.apply(s2, mt.ASSISTANT, {"message": "", "move": {"type": "accept"}})
    # A mutual slot can be accepted, and both sides score their own value.
    slot = env.ground_truth(inst)["mutual_slots"][0]
    s3, _ = env.apply(s2, mt.ASSISTANT, {"message": "", "move": {"type": "propose", "slot": slot}})
    s4, _ = env.apply(s3, mt.CONTACT, {"message": "ok", "move": {"type": "accept"}})
    assert env.to_act(s4) is None and s4.slot == slot
    assert env.outcome(s4) == {mt.ASSISTANT: inst.free[mt.ASSISTANT][slot],
                               mt.CONTACT: inst.free[mt.CONTACT][slot]}


def test_private_calendars_stay_private():
    env, s = _state()
    obs = env.observe(s, mt.ASSISTANT)
    text = json.dumps(obs)
    assert {k for k, v in obs["calendar"].items() if v is not None} == set(s.inst.free[mt.ASSISTANT])
    assert "contact" not in text.replace('"role": "assistant"', "")
    system, user = mt.MeetingText().render(obs)
    assert "busy" in system and "Your free slots" in system and "Respond with a JSON object" in user


def test_deadline_and_fallback():
    env, s = _state()
    for _ in range(s.inst.max_turns):
        seat = env.to_act(s)
        s, _ = env.apply(s, seat, env.fallback_action(s, seat))
    assert s.end_reason == "deadline" and env.outcome(s) == {mt.ASSISTANT: 0.0, mt.CONTACT: 0.0}


def test_text_parse():
    text = mt.MeetingText()
    action, _ = text.parse(json.dumps({"message": "hi", "move": {"type": "propose", "slot": "Mon 09:00",
                                                                  "extra": 1}}))
    assert action == {"message": "hi", "move": {"type": "propose", "slot": "Mon 09:00"}}
    assert text.parse({"message": "x", "move": {"type": "accept", "slot": "Mon 09:00"}})[0]["move"] == {
        "type": "accept"}
    for bad in ("not json", {"move": "x"}, {"move": {"type": "fly"}}):
        with pytest.raises(MalformedOutput):
            text.parse(bad)


def test_policies_differ_and_unknown_is_rejected():
    env, s = _state(5)
    obs = env.observe(s, mt.CONTACT)
    moves = {name: mt.make_policy(name, mt.CONTACT).decide(obs)["move"] for name in mt.POLICIES}
    assert all(m["type"] == "propose" for m in moves.values())
    with pytest.raises(ValueError):
        mt.make_policy("nope", mt.CONTACT)


def test_canary_runs_verifies_and_reports(tmp_path):
    asyncio.run(run_matched(load_config(ROOT / "configs" / "meeting-canary.yaml"), tmp_path))
    entry = envs.get("meeting_scheduling")
    traces = sorted((tmp_path / "traces").glob("*.jsonl"))
    assert len(traces) == 36
    for t in traces:
        assert verify(entry.env(), read_trace(t), make_policy=entry.make_policy) == []
    report = build_report(tmp_path)
    assert report["cells"] == {"done": 36}
    deals = [entry.episode_metrics(read_trace(t))["deal"] for t in traces]
    assert 0 < sum(deals) < len(deals)  # scripted bots reach deals, but not always


def test_runs_through_the_harbor_sidecar():
    pytest.importorskip("mcp")
    from parley_sim.adapters.harbor import CHAT_NOTE, HarborRuntime, RuntimeConfig, harbor_cells
    from parley_sim.matched import build_grid

    cfg = load_config(ROOT / "configs" / "meeting-canary.yaml")
    cell = next(c for c in harbor_cells(build_grid(cfg), run_id=cfg.run_id, env=cfg.env,
                                        env_config=cfg.env_config, lineup=cfg.lineup)
                if c.arm == "soc")
    rt = HarborRuntime(RuntimeConfig(cell=cell, token="t" * 24))
    text = asyncio.run(rt.start())
    assert "Your free slots" in text and "take_turn" in text
    assert CHAT_NOTE not in text  # only when the other side is played from outside
    best = max(rt.episode.state.inst.free[mt.ASSISTANT].items(), key=lambda kv: kv[1])[0]
    for _ in range(10):
        if "Do not call any more tools" in text:
            break
        text = asyncio.run(rt.take_turn({"message": "How about this?",
                                         "move": {"type": "propose", "slot": best}}))
    result = rt.result()
    assert result["completed"] == 1 and result["verified"] == 1 and "no_deal" in result


def test_chat_leak_counts_stated_ratings_only():
    lines = [{"type": "decision", "seat": mt.ASSISTANT,
              "observation": {"calendar": {"Mon 09:00": 8, "Thu 11:00": 10, "Wed 13:00": 2,
                                           "Tue 10:00": None}}}]
    for text in ("I countered with Thu 11:00 (worth 10 to my manager).", "Try Mon 09:00 (8).",
                 "Thu 11:00 = 10 for me.", "Thursday 11:00, which my manager rates 10.",
                 "Monday 09:00 (value 8 for my manager).", "Thu 11:00 is a 10/10."):
        assert mt.chat_leak(lines, mt.ASSISTANT, text), text
    for text in ("Done, your turn.", "Propose Wed 12:00 (best).", "Thu 11:00 works for us.",
                 "Counter with a high-value slot.", "Mon 09:00 or Wed 13:00?", "worth 5 to me",
                 "I have 3 options."):
        assert not mt.chat_leak(lines, mt.ASSISTANT, text), text
    assert not mt.chat_leak(lines, mt.CONTACT, "Thu 11:00 = 10 for me.")  # not that seat's values


def test_accept_books_the_standing_proposal_and_says_so():
    env, s = _state()
    slot = env.ground_truth(s.inst)["mutual_slots"][0]
    s, _ = env.apply(s, mt.CONTACT, {"message": "If not, Friday works too.",
                                     "move": {"type": "propose", "slot": slot}})
    s, _ = env.apply(s, mt.ASSISTANT, {"message": "Friday it is!", "move": {"type": "accept"}})
    assert s.slot == slot
    assert env.describe_end(s, mt.ASSISTANT) == (
        f"Booked: {slot} (value to you: {s.inst.free[mt.ASSISTANT][slot]:g}).")
    system, _ = mt.MeetingText().render(env.observe(_state()[1], mt.ASSISTANT))
    assert "To book a slot they only mentioned in a message, `propose` it." in system


def test_calendar_density_is_bounded():
    env = mt.MeetingScheduling()
    for busy in (1.0, -0.1, 0.9):
        with pytest.raises(ValueError, match="busy must be"):
            env.generate(1, {"busy": busy})
    assert len(env.ground_truth(env.generate(1, {"busy": mt.MAX_BUSY}))["mutual_slots"]) >= 4
