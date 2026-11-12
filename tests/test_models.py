import asyncio
import json

import pytest

from parley_sim.drivers.models import (
    Breaker,
    EpisodeAbort,
    ModelSeat,
    ModelSpec,
    ProviderHalted,
    ProviderRefusal,
)
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


def status_error(code):
    err = RuntimeError(f"Error code: {code}")
    err.status_code = code
    return err


@pytest.mark.parametrize("code", [401, 402, 403, 404])
def test_account_and_model_errors_halt_the_run(code):
    breaker = Breaker()
    spec = ModelSpec(name="m", provider="openai_compat", model="x")
    s = ModelSeat(spec, SupportDeskText(), FakeClient([status_error(code)]), breaker)
    with pytest.raises(ProviderHalted, match=str(code)):
        asyncio.run(s.decide(request()))
    assert breaker.tripped is not None


@pytest.mark.parametrize("code", [400, 422])
def test_rejected_request_aborts_the_episode(code):
    breaker = Breaker()
    spec = ModelSpec(name="m", provider="openai_compat", model="x")
    s = ModelSeat(spec, SupportDeskText(), FakeClient([status_error(code)]), breaker)
    with pytest.raises(EpisodeAbort, match=f"m at seat customer.*{code}") as info:
        asyncio.run(s.decide(request()))
    assert info.value.retryable is False
    assert breaker.tripped is None  # one bad request does not halt the run


def test_consecutive_failed_turns_abort_the_episode():
    # Timeouts and provider errors count toward the same guard; a turn that
    # got any reply resets it.
    client = FakeClient([status_error(429), GOOD, status_error(429), status_error(503),
                         status_error(429)])
    s = seat(client)
    assert asyncio.run(s.decide(request())).strike == "exception"
    assert asyncio.run(s.decide(request())).strike is None
    assert asyncio.run(s.decide(request())).strike == "exception"
    assert asyncio.run(s.decide(request())).strike == "exception"
    with pytest.raises(EpisodeAbort, match=r"m at seat customer: 3 turns in a row.*429") as info:
        asyncio.run(s.decide(request()))
    assert info.value.retryable is True


def test_timeouts_count_toward_the_failure_guard():
    # Timed-out calls are cancelled before FakeClient takes a reply.
    s = seat(FakeClient([RuntimeError("connection reset")], delay=0.2), timeout_s=0.05)
    assert asyncio.run(s.decide(request())).strike == "timeout"
    assert asyncio.run(s.decide(request())).strike == "timeout"
    s.spec = s.spec.model_copy(update={"timeout_s": 5.0})
    with pytest.raises(EpisodeAbort, match="connection reset"):
        asyncio.run(s.decide(request()))


def test_failed_calls_are_recorded_as_attempts_without_usage():
    res = asyncio.run(seat(FakeClient([RuntimeError("503")])).decide(request()))
    assert len(res.raw["attempts"]) == 1 and "503" in res.raw["attempts"][0]["error"]
    assert "latency_s" in res.raw["attempts"][0] and "usage" not in res.raw["attempts"][0]
    assert res.raw["usage"] == {"input_tokens": 0, "output_tokens": 0}
    # A failed retry after a billed malformed reply keeps the billed tokens.
    res = asyncio.run(seat(FakeClient(["nope", RuntimeError("503")])).decide(request()))
    assert res.strike == "exception" and len(res.raw["attempts"]) == 2
    assert res.raw["usage"] == {"input_tokens": 10, "output_tokens": 5}
    res = asyncio.run(seat(FakeClient([GOOD], delay=0.2), timeout_s=0.05).decide(request()))
    assert [set(a) for a in res.raw["attempts"]] == [{"error", "latency_s"}]


def test_refused_calls_keep_their_usage():
    class Refuser:
        async def complete(self, *, system, user, schema):
            raise ProviderRefusal("cyber", raw={"served_model": "m", "stop_reason": "end_turn",
                                                "usage": {"input_tokens": 30, "output_tokens": 2,
                                                          "cost_usd": 0.01}})

    res = asyncio.run(seat(Refuser()).decide(request()))
    assert res.strike == "refusal"
    assert res.raw["usage"] == {"input_tokens": 30, "output_tokens": 2, "cost_usd": 0.01}
    assert res.raw["attempts"][0]["stop_reason"] == "refusal"


@pytest.mark.parametrize("empty", [None, "", "  \n"])
def test_conversation_never_records_an_empty_reply(empty):
    env = SupportDesk()
    state = env.initial_state(env.generate(0, {}))
    client = HistoryClient([empty, GOOD])
    spec = ModelSpec(name="m", provider="openai_compat", model="x", conversation=True)
    s = ModelSeat(spec, SupportDeskText(), client)
    first = env.observe(state, "customer")
    res = asyncio.run(s.decide(DecisionRequest("0:customer", "customer", 0, first, {})))
    assert res.strike is None and len(res.raw["attempts"]) == 2
    # The retry is the same prompt with the correction appended, sent with no
    # exchange for the empty reply; only the answered prompt enters history.
    assert client.histories == [[], []]
    assert client.users[1].startswith(client.users[0])
    assert "Your previous reply was invalid (empty reply)" in client.users[1]
    assert s.history == [{"user": client.users[1], "assistant": GOOD}]


def test_empty_replies_do_not_move_the_conversation_start_point():
    env = SupportDesk()
    state = env.initial_state(env.generate(0, {}))
    client = HistoryClient([GOOD, None, "", GOOD])
    spec = ModelSpec(name="m", provider="openai_compat", model="x", conversation=True)
    s = ModelSeat(spec, SupportDeskText(), client)
    first = env.observe(state, "customer")
    asyncio.run(s.decide(DecisionRequest("0:customer", "customer", 0, first, {})))
    state, _ = env.apply(state, "customer", {"message": "hi", "move": {"type": "none"}})
    state, _ = env.apply(state, "agent", {"message": "Stay with us", "move": {"type": "none"}})
    second = env.observe(state, "customer")
    res = asyncio.run(s.decide(DecisionRequest("2:customer", "customer", 2, second, {})))
    assert res.strike == "malformed_output" and len(s.history) == 1
    assert s._previous == first
    state, _ = env.apply(state, "customer", {"message": "hm", "move": {"type": "none"}})
    state, _ = env.apply(state, "agent", {"message": "Last offer", "move": {"type": "none"}})
    third = env.observe(state, "customer")
    asyncio.run(s.decide(DecisionRequest("4:customer", "customer", 4, third, {})))
    # The third turn's update still covers everything since the last real reply.
    assert "Stay with us" in client.users[3] and "Last offer" in client.users[3]
    assert client.histories[3] == [{"user": client.users[0], "assistant": GOOD}]


def _api_cfg():
    from parley_sim.cli import smoke_config
    from parley_sim.matched import MatchedConfig
    base = smoke_config("0:2").model_dump()
    base["lineup"].append({"name": "api-customer", "provider": "openai_compat", "model": "x"})
    return MatchedConfig(**{**base, "adaptive_counterparts": ["api-customer"], "concurrency": 1})


class Failing:
    def __init__(self, code):
        self.code = code

    async def complete(self, *, system, user, schema):
        raise status_error(self.code)


@pytest.mark.parametrize("code,message", [(400, "provider rejected the request"),
                                          (429, "3 turns in a row failed")])
def test_matched_records_a_failing_model_cell_as_errored(tmp_path, code, message):
    from parley_sim.matched import run_matched
    from parley_sim.metrics import load_manifest
    from parley_sim.trace import read_trace

    asyncio.run(run_matched(_api_cfg(), tmp_path, client_factory=lambda spec: Failing(code)))
    _, cells = load_manifest(tmp_path)
    assert {(r["arm"], r["status"]) for r in cells} == {("soc", "done"), ("aoc", "errored")}
    for r in cells:
        if r["status"] == "errored":
            assert "EpisodeAbort" in r["error"] and message in r["error"]
            assert read_trace(tmp_path / r["trace"])[-1]["status"] == "errored"


@pytest.fixture
def fake_usersim(monkeypatch):
    """UserSimPersonaText with a stand-in persona block (usersim is optional)."""
    from parley_sim.drivers import usersim_persona
    monkeypatch.setattr(usersim_persona, "usersim_block",
                        lambda persona, locale="en_US": ("PERSONA BLOCK",
                                                         {"usersim_version": "0-test"}))
    return usersim_persona.UserSimPersonaText


def _two_turns(text):
    """Play two conversation turns of the customer seat; return the client."""
    env = SupportDesk()
    state = env.initial_state(env.generate(0, {}))
    client = HistoryClient([GOOD, GOOD])
    spec = ModelSpec(name="m", provider="openai_compat", model="x", conversation=True)
    s = ModelSeat(spec, text, client)
    first = env.observe(state, "customer")
    asyncio.run(s.decide(DecisionRequest("0:customer", "customer", 0, first, {})))
    state, _ = env.apply(state, "customer", {"message": "hi", "move": {"type": "none"}})
    state, _ = env.apply(state, "agent", {"message": "Stay with us", "move": {"type": "none"}})
    asyncio.run(s.decide(DecisionRequest("2:customer", "customer", 2,
                                         env.observe(state, "customer"), {})))
    return client


def test_persona_seat_in_conversation_sends_only_updates(fake_usersim):
    base = SupportDeskText()
    text = fake_usersim(base, {"first_name": "Dana"})
    assert text.render_update.__func__ is type(base).render_update
    client = _two_turns(text)
    assert client.users == _two_turns(SupportDeskText()).users  # same as unwrapped
    assert "Conversation so far" not in client.users[1]


def test_persona_wrapper_without_base_update_resends_full_prompts(fake_usersim):
    class NoUpdate:
        """SupportDeskText's interface minus render_update."""

        def __init__(self):
            self._inner = SupportDeskText()
            self.prompt_version = self._inner.prompt_version
            self.action_schema = self._inner.action_schema

        def render(self, observation):
            return self._inner.render(observation)

        def parse(self, payload):
            return self._inner.parse(payload)

    text = fake_usersim(NoUpdate(), {"first_name": "Dana"})
    assert not hasattr(text, "render_update")
    client = _two_turns(text)
    assert client.users == _two_turns(NoUpdate()).users
    assert "Conversation so far" in client.users[1]  # the full prompt again
