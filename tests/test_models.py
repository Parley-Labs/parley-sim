import asyncio
import json

from parley_sim.drivers.models import ModelSeat, ModelSpec
from parley_sim.envs.support_desk import SupportDesk, SupportDeskText
from parley_sim.runtime import DecisionRequest

GOOD = json.dumps({"message": "hi", "move": {"type": "none"}, "explanation": "r"})


class FakeClient:
    def __init__(self, replies, delay=0.0):
        self.replies = list(replies)
        self.delay = delay
        self.users = []

    async def complete(self, *, system, user, schema):
        self.users.append(user)
        await asyncio.sleep(self.delay)
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply, {"usage": {"input_tokens": 10, "output_tokens": 5}}


def request():
    env = SupportDesk()
    state = env.initial_state(env.generate(0, {}))
    return DecisionRequest("0:customer", "customer", 0, env.observe(state, "customer"), {})


def seat(client, timeout_s=5.0):
    spec = ModelSpec(name="m", provider="openai_compat", model="x", timeout_s=timeout_s)
    return ModelSeat(spec, SupportDeskText(), client)


def test_valid_reply():
    res = asyncio.run(seat(FakeClient([GOOD])).decide(request()))
    assert res.strike is None and res.action["move"] == {"type": "none"}
    assert res.explanation == "r" and res.raw["usage"] == {"input_tokens": 10, "output_tokens": 5}


def test_one_repair_retry_then_success():
    client = FakeClient(["nope", GOOD])
    res = asyncio.run(seat(client).decide(request()))
    assert res.strike is None and len(res.raw["attempts"]) == 2
    assert "previous reply was invalid" in client.users[1]
    assert res.raw["usage"]["input_tokens"] == 20


def test_malformed_twice_is_a_strike():
    res = asyncio.run(seat(FakeClient(["nope", "{}"])).decide(request()))
    assert res.action is None and res.strike == "malformed_output"


def test_timeout_and_exception_are_strikes():
    res = asyncio.run(seat(FakeClient([GOOD], delay=0.2), timeout_s=0.05).decide(request()))
    assert res.strike == "timeout"
    res = asyncio.run(seat(FakeClient([RuntimeError("503")])).decide(request()))
    assert res.strike == "exception" and "503" in res.detail


class HistoryClient(FakeClient):
    def __init__(self, replies):
        super().__init__(replies)
        self.histories = []

    async def complete(self, *, system, user, schema, history=None):
        self.histories.append(history)
        return await super().complete(system=system, user=user, schema=schema)


def test_conversation_seat_sends_only_what_changed_after_its_own_replies():
    env = SupportDesk()
    state = env.initial_state(env.generate(0, {}))
    client = HistoryClient([GOOD, "nope", GOOD])
    spec = ModelSpec(name="m", provider="openai_compat", model="x", conversation=True)
    s = ModelSeat(spec, SupportDeskText(), client)
    first = env.observe(state, "customer")
    asyncio.run(s.decide(DecisionRequest("0:customer", "customer", 0, first, {})))
    state, _ = env.apply(state, "customer", {"message": "hi", "move": {"type": "none"}})
    state, _ = env.apply(state, "agent", {"message": "Stay with us", "move": {"type": "none"}})
    second = env.observe(state, "customer")
    res = asyncio.run(s.decide(DecisionRequest("2:customer", "customer", 2, second, {})))
    assert res.strike is None
    # Turn 1: the full prompt, no history. Turn 2: only the agent's new message,
    # after turn 1's exchange. The repair retry keeps the invalid reply and adds a
    # correction instead of resending the prompt.
    assert client.histories[0] == [] and "You speak first." in client.users[0]
    assert client.histories[1] == [{"user": client.users[0], "assistant": GOOD}]
    assert "[Counterpart] \"Stay with us\"" in client.users[1]
    assert "Conversation so far" not in client.users[1] and "[You]" not in client.users[1]
    assert client.histories[2][-1] == {"user": client.users[1], "assistant": "nope"}
    assert client.users[2].startswith("Your previous reply was invalid")


def test_stateless_seat_is_unchanged():
    client = FakeClient([GOOD])
    asyncio.run(seat(client).decide(request()))  # FakeClient takes no history argument


def test_update_restates_own_move_only_when_the_record_differs():
    env = SupportDesk()
    text = SupportDeskText()
    state = env.initial_state(env.generate(0, {}))
    state, _ = env.apply(state, "customer", {"message": "hi", "move": {"type": "none"}})
    before = env.observe(state, "agent")
    sent = {"message": "ok", "move": {"type": "none"}}
    state, _ = env.apply(state, "agent", sent)
    state, _ = env.apply(state, "customer", {"message": "still here", "move": {"type": "none"}})
    after = env.observe(state, "agent")
    assert "[You]" not in text.render_update(before, after, sent)
    assert "Your standing proposal: none" in text.render_update(before, after, sent)
    other = {"message": "different", "move": {"type": "none"}}
    assert "could not be applied as sent" in text.render_update(before, after, other)
    assert "no usable reply" in text.render_update(before, after, None)


def test_update_reports_a_truncated_message():
    from parley_sim.envs.support_desk.env import MAX_MESSAGE_CHARS
    env, text = SupportDesk(), SupportDeskText()
    state = env.initial_state(env.generate(0, {}))
    state, _ = env.apply(state, "customer", {"message": "hi", "move": {"type": "none"}})
    before = env.observe(state, "agent")
    sent = {"message": "x" * (MAX_MESSAGE_CHARS + 50), "move": {"type": "none"}}
    state, _ = env.apply(state, "agent", sent)
    state, _ = env.apply(state, "customer", {"message": "ok", "move": {"type": "none"}})
    assert "could not be applied as sent" in text.render_update(before, env.observe(state, "agent"), sent)
