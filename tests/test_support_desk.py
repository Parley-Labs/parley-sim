import pytest

from parley_sim.core.protocol import IllegalAction
from parley_sim.envs.support_desk import SupportDesk, SupportDeskText, make_policy
from parley_sim.envs.support_desk import scenario as sc
from parley_sim.core.protocol import MalformedOutput


@pytest.mark.parametrize("seed", range(50))
def test_every_scenario_has_a_mutually_beneficial_deal(seed):
    inst = sc.generate(seed, {})
    gt = sc.ground_truth(inst)
    assert gt["n_mutually_beneficial"] >= 1
    nash = sc.deal_from_json(gt["nash_deal"])
    assert sc.authorized(inst, nash)
    assert any(f["deal"] == gt["nash_deal"] for f in gt["frontier"])
    assert gt["max_joint_surplus"] >= sum(gt["nash_surplus"].values())


def test_generate_is_deterministic_and_validates_config():
    assert sc.generate(5, {}) == sc.generate(5, {})
    assert sc.generate(5, {}) != sc.generate(6, {})
    with pytest.raises(ValueError):
        sc.generate(5, {"max_turn": 4})
    with pytest.raises(ValueError):
        sc.generate(5, {"max_turns": 1})


def test_observation_hides_counterpart_private_information():
    env = SupportDesk()
    state = env.initial_state(env.generate(0, {}))
    cust = env.observe(state, "customer")
    agent = env.observe(state, "agent")
    assert "authorization_limits" not in cust["private"]
    assert "authorization_limits" in agent["private"]
    assert cust["private"]["option_values"] != agent["private"]["option_values"]


def test_turn_order_and_out_of_turns():
    env = SupportDesk()
    state = env.initial_state(env.generate(0, {"max_turns": 2}))
    assert env.to_act(state) == "customer"
    with pytest.raises(IllegalAction):
        env.apply(state, "agent", {"message": "", "move": {"type": "none"}})
    state, _ = env.apply(state, "customer", {"message": "hi", "move": {"type": "none"}})
    assert env.to_act(state) == "agent"
    state, events = env.apply(state, "agent", {"message": "", "move": {"type": "none"}})
    assert state.ended and state.end_reason == "max_turns" and env.to_act(state) is None
    assert env.outcome(state) == {"agent": 0.0, "customer": float(state.inst.customer_walkaway)}


def test_accept_requires_a_standing_offer_and_respects_limits():
    env = SupportDesk()
    inst = env.generate(0, {})
    state = env.initial_state(inst)
    with pytest.raises(IllegalAction):
        env.apply(state, "customer", {"message": "", "move": {"type": "accept"}})
    greedy = {"discount_pct": 40, "credit_usd": 100, "upgrade": "premium", "term": "monthly"}
    state, _ = env.apply(state, "customer",
                         {"message": "", "move": {"type": "propose", "deal": greedy}})
    if not sc.authorized(inst, sc.deal_from_json(greedy)):
        with pytest.raises(IllegalAction):
            env.apply(state, "agent", {"message": "", "move": {"type": "accept"}})


def test_bad_deal_is_illegal():
    env = SupportDesk()
    state = env.initial_state(env.generate(0, {}))
    with pytest.raises(IllegalAction):
        env.apply(state, "customer", {"message": "", "move": {"type": "propose",
                                                               "deal": {"discount_pct": 15}}})


def test_policy_seat_mismatch_is_rejected():
    with pytest.raises(ValueError):
        make_policy("hardliner", "agent")
    with pytest.raises(ValueError):
        make_policy("nope", "customer")


def test_text_interface_round_trip():
    env = SupportDesk()
    state = env.initial_state(env.generate(0, {}))
    text = SupportDeskText()
    system, user = text.render(env.observe(state, "customer"))
    assert "switching" in system and "You speak first." in user
    action, explanation = text.parse('{"explanation": "r", "message": "m", '
                                   '"move": {"type": "propose", "deal": {"discount_pct": 10, '
                                   '"credit_usd": 0, "upgrade": "none", "term": "monthly"}}}')
    assert explanation == "r"
    assert "reasoning" not in text.action_schema["properties"]
    env.apply(state, "customer", action)
    with pytest.raises(MalformedOutput):
        text.parse("not json")
    with pytest.raises(MalformedOutput):
        text.parse({"message": "m"})
