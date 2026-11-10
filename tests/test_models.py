import asyncio
import json

from parley_sim.drivers.models import ModelSeat, ModelSpec
from parley_sim.envs.support_desk import SupportDesk, SupportDeskText
from parley_sim.runtime import DecisionRequest

GOOD = json.dumps({"reasoning": "r", "message": "hi", "move": {"type": "none"}})


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
    assert res.reasoning == "r" and res.raw["usage"] == {"input_tokens": 10, "output_tokens": 5}


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
