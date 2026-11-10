import asyncio
from types import SimpleNamespace as NS

import pytest

from parley_sim.drivers.models import AnthropicClient, ModelSpec, OpenAIClient

pytest.importorskip("anthropic")
pytest.importorskip("openai")


class Recorder:
    def __init__(self, response):
        self.response, self.kwargs = response, None

    async def create(self, **kwargs):
        self.kwargs = kwargs
        return self.response


def test_anthropic_uses_structured_output_and_omits_null_temperature(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "x")
    client = AnthropicClient(ModelSpec(name="a", provider="anthropic", model="m", effort="low",
                                       provider_params={"metadata": {"user_id": "u"}}))
    resp = NS(model="m-served", stop_reason="end_turn", usage=NS(input_tokens=7, output_tokens=3),
              content=[NS(type="thinking", thinking="hmm"),
                       NS(type="text", text='{"move": {"type": "none"}}')])
    rec = Recorder(resp)
    client._client = NS(messages=rec)
    payload, raw = asyncio.run(client.complete(system="s", user="u", schema={"type": "object"}))
    assert payload == '{"move": {"type": "none"}}'
    assert raw["usage"] == {"input_tokens": 7, "output_tokens": 3}
    assert raw["provider_reasoning"] == "hmm"
    assert "temperature" not in rec.kwargs and "tool_choice" not in rec.kwargs
    assert rec.kwargs["output_config"] == {"format": {"type": "json_schema",
                                                      "schema": {"type": "object"}},
                                           "effort": "low"}
    assert rec.kwargs["metadata"] == {"user_id": "u"}


def test_anthropic_refusal_becomes_a_refusal_strike(monkeypatch):
    from parley_sim.drivers.models import ModelSeat
    from parley_sim.envs.support_desk import SupportDesk, SupportDeskText
    from parley_sim.runtime import DecisionRequest

    monkeypatch.setenv("ANTHROPIC_API_KEY", "x")
    spec = ModelSpec(name="a", provider="anthropic", model="m")
    client = AnthropicClient(spec)
    client._client = NS(messages=Recorder(NS(stop_reason="refusal", content=[],
                                             stop_details=NS(category="cyber"))))
    env = SupportDesk()
    state = env.initial_state(env.generate(0, {}))
    req = DecisionRequest("0:customer", "customer", 0, env.observe(state, "customer"), {})
    res = asyncio.run(ModelSeat(spec, SupportDeskText(), client).decide(req))
    assert res.strike == "refusal" and res.detail == "cyber"


@pytest.mark.parametrize("provider,tokens_key", [("openai", "max_completion_tokens"),
                                                 ("openai_compat", "max_tokens")])
def test_openai_paths(monkeypatch, provider, tokens_key):
    monkeypatch.setenv("OPENAI_API_KEY", "x")
    monkeypatch.setenv("OPENROUTER_API_KEY", "x")
    client = OpenAIClient(ModelSpec(name="o", provider=provider, model="m", temperature=0.2,
                                    base_url="https://example.invalid/v1"))
    msg = NS(content='{"move": {"type": "none"}}', reasoning="why")
    resp = NS(model="m", choices=[NS(message=msg, finish_reason="stop")],
              usage=NS(prompt_tokens=11, completion_tokens=4))
    rec = Recorder(resp)
    client._client = NS(chat=NS(completions=rec))
    payload, raw = asyncio.run(client.complete(system="s", user="u", schema={"type": "object"}))
    assert payload == '{"move": {"type": "none"}}'
    assert raw["usage"] == {"input_tokens": 11, "output_tokens": 4}
    assert rec.kwargs[tokens_key] == 4096 and rec.kwargs["temperature"] == 0.2
    assert raw["provider_reasoning"] == "why"


def test_openrouter_effort_merges_into_extra_body_and_cost_is_recorded(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "placeholder")
    routing = {"provider": {"order": ["Anthropic"], "allow_fallbacks": False},
               "reasoning": {"exclude": False}}
    client = OpenAIClient(ModelSpec(name="o", provider="openai_compat", model="m", effort="low",
                                    base_url="https://openrouter.ai/api/v1",
                                    provider_params={"extra_body": routing}))
    msg = NS(content="{}", reasoning=None)
    resp = NS(model="m", choices=[NS(message=msg, finish_reason="stop")],
              usage=NS(prompt_tokens=1, completion_tokens=2, cost=0.0123))
    rec = Recorder(resp)
    client._client = NS(chat=NS(completions=rec))
    _, raw = asyncio.run(client.complete(system="s", user="u", schema={}))
    assert rec.kwargs["extra_body"]["reasoning"] == {"exclude": False, "effort": "low"}
    assert rec.kwargs["extra_body"]["provider"] == routing["provider"]
    assert "reasoning_effort" not in rec.kwargs
    assert raw["usage"]["cost_usd"] == 0.0123


def test_missing_gateway_key_never_falls_back_to_openai_key(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "secret")
    with pytest.raises(RuntimeError, match="OPENROUTER_API_KEY"):
        OpenAIClient(ModelSpec(name="o", provider="openai_compat", model="m"))
