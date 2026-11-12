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


GATEWAY = "https://gateway.invalid/v1"


@pytest.mark.parametrize("provider,tokens_key", [("openai", "max_completion_tokens"),
                                                 ("openai_compat", "max_tokens")])
def test_openai_paths(monkeypatch, provider, tokens_key):
    monkeypatch.setenv("OPENAI_API_KEY", "x")
    monkeypatch.setenv("OPENROUTER_API_KEY", "x")
    client = OpenAIClient(ModelSpec(name="o", provider=provider, model="m", temperature=0.2,
                                    base_url="https://example.invalid/v1",
                                    api_key_env="OPENAI_API_KEY" if provider == "openai" else None))
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
        OpenAIClient(ModelSpec(name="o", provider="openai_compat", model="m", base_url=GATEWAY))


def test_keys_only_reach_their_own_host(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "secret")
    monkeypatch.setenv("OPENROUTER_API_KEY", "placeholder")
    with pytest.raises(ValueError, match="api_key_env"):
        OpenAIClient(ModelSpec(name="o", provider="openai", model="m", base_url=GATEWAY))
    with pytest.raises(ValueError, match="needs base_url"):
        OpenAIClient(ModelSpec(name="o", provider="openai_compat", model="m"))


def test_openai_structured_output_refusal_is_a_refusal(monkeypatch):
    from parley_sim.drivers.models import ProviderRefusal
    monkeypatch.setenv("OPENROUTER_API_KEY", "placeholder")
    client = OpenAIClient(ModelSpec(name="o", provider="openai_compat", model="m", base_url=GATEWAY))
    msg = NS(content=None, refusal="I can't help with that.")
    resp = NS(model="m", choices=[NS(message=msg, finish_reason="stop")],
              usage=NS(prompt_tokens=1, completion_tokens=0))
    client._client = NS(chat=NS(completions=Recorder(resp)))
    with pytest.raises(ProviderRefusal):
        asyncio.run(client.complete(system="s", user="u", schema={}))


def test_openai_content_filter_is_a_refusal(monkeypatch):
    from parley_sim.drivers.models import ProviderRefusal
    monkeypatch.setenv("OPENROUTER_API_KEY", "placeholder")
    client = OpenAIClient(ModelSpec(name="o", provider="openai_compat", model="m", base_url=GATEWAY))
    resp = NS(model="m", choices=[NS(message=NS(content=None), finish_reason="content_filter")],
              usage=NS(prompt_tokens=1, completion_tokens=0))
    client._client = NS(chat=NS(completions=Recorder(resp)))
    with pytest.raises(ProviderRefusal):
        asyncio.run(client.complete(system="s", user="u", schema={}))


def test_anthropic_requests_summarized_thinking(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "x")
    client = AnthropicClient(ModelSpec(name="a", provider="anthropic", model="m"))
    resp = NS(model="m", stop_reason="end_turn", usage=NS(input_tokens=1, output_tokens=1),
              content=[NS(type="thinking", thinking="plan"), NS(type="text", text="{}")])
    rec = Recorder(resp)
    client._client = NS(messages=rec)
    _, raw = asyncio.run(client.complete(system="s", user="u", schema={}))
    assert rec.kwargs["thinking"] == {"type": "adaptive", "display": "summarized"}
    assert raw["provider_reasoning"] == "plan"


def test_custom_endpoints_need_an_explicit_key_variable(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "secret")
    monkeypatch.setenv("OPENAI_API_KEY", "secret")
    with pytest.raises(ValueError, match="api_key_env"):
        AnthropicClient(ModelSpec(name="a", provider="anthropic", model="m", base_url=GATEWAY))
    monkeypatch.setenv("OPENAI_BASE_URL", GATEWAY)  # the SDK would send the key here
    with pytest.raises(ValueError, match="api_key_env"):
        OpenAIClient(ModelSpec(name="o", provider="openai", model="m"))
    monkeypatch.setenv("OPENAI_BASE_URL", "")
    OpenAIClient(ModelSpec(name="o", provider="openai", model="m"))  # default endpoint is fine


def test_keys_travel_only_over_tls_except_to_localhost(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "placeholder")
    with pytest.raises(ValueError, match="https"):
        OpenAIClient(ModelSpec(name="o", provider="openai_compat", model="m",
                               base_url="http://gateway.example/v1"))
    OpenAIClient(ModelSpec(name="o", provider="openai_compat", model="m",
                           base_url="http://localhost:8000/v1"))


def test_clients_replay_conversation_history(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "placeholder")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "x")
    history = [{"user": "u1", "assistant": "a1"}]
    oa = OpenAIClient(ModelSpec(name="o", provider="openai_compat", model="m", base_url=GATEWAY))
    rec = Recorder(NS(model="m", choices=[NS(message=NS(content="{}", refusal=None), finish_reason="stop")],
                      usage=NS(prompt_tokens=1, completion_tokens=1)))
    oa._client = NS(chat=NS(completions=rec))
    asyncio.run(oa.complete(system="s", user="u2", schema={}, history=history))
    assert [m["content"] for m in rec.kwargs["messages"]] == ["s", "u1", "a1", "u2"]
    an = AnthropicClient(ModelSpec(name="a", provider="anthropic", model="m"))
    rec = Recorder(NS(model="m", stop_reason="end_turn", usage=NS(input_tokens=1, output_tokens=1),
                      content=[NS(type="text", text="{}")]))
    an._client = NS(messages=rec)
    asyncio.run(an.complete(system="s", user="u2", schema={}, history=history))
    assert [m["role"] for m in rec.kwargs["messages"]] == ["user", "assistant", "user"]
