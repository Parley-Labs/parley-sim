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


def test_anthropic_forces_tool_and_omits_null_temperature(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "x")
    client = AnthropicClient(ModelSpec(name="a", provider="anthropic", model="m",
                                       provider_params={"top_k": 5}))
    resp = NS(model="m-served", stop_reason="tool_use", usage=NS(input_tokens=7, output_tokens=3),
              content=[NS(type="thinking", thinking="hmm"),
                       NS(type="tool_use", input={"move": {"type": "none"}})])
    rec = Recorder(resp)
    client._client = NS(messages=rec)
    payload, raw = asyncio.run(client.complete(system="s", user="u", schema={"type": "object"}))
    assert payload == {"move": {"type": "none"}}
    assert raw["usage"] == {"input_tokens": 7, "output_tokens": 3}
    assert raw["provider_reasoning"] == "hmm"
    assert "temperature" not in rec.kwargs and rec.kwargs["top_k"] == 5
    assert rec.kwargs["tool_choice"] == {"type": "tool", "name": "take_turn"}


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
