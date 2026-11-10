"""Model seats: answer a DecisionRequest by rendering it for a model, calling a
provider, and parsing the reply (one repair retry on malformed output)."""
from __future__ import annotations

import asyncio
import json
import os
import time
from typing import Any, Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field, model_validator

from parley_sim.core.protocol import JSON, MalformedOutput, Policy, TextInterface
from parley_sim.runtime.episode import DecisionRequest, DecisionResult

Provider = Literal["anthropic", "openai", "openai_compat", "scripted"]


class ModelSpec(BaseModel):
    """One lineup entry. `temperature=None` omits the parameter entirely (some
    reasoning models reject any explicit value). `provider_params` is merged
    into the request last and recorded, so tuning is a config change. The
    `scripted` provider plays a named policy through the text interface, for
    offline runs and tests."""
    model_config = ConfigDict(extra="forbid")

    name: str = Field(pattern=r"^[A-Za-z0-9._-]+$")
    provider: Provider
    model: str
    base_url: str | None = None
    api_key_env: str | None = None
    temperature: float | None = None
    effort: Literal["low", "medium", "high", "xhigh", "max"] | None = None
    max_tokens: int = 4096
    timeout_s: float = 120.0
    provider_params: dict[str, Any] | None = None

    @model_validator(mode="after")
    def _anthropic_has_no_sampling(self) -> "ModelSpec":
        if self.provider == "anthropic" and self.temperature is not None:
            raise ValueError("the Anthropic API no longer accepts temperature; remove it")
        return self


class ProviderRefusal(Exception):
    """The provider declined to answer (a safety stop, not malformed output)."""


class ProviderClient(Protocol):
    async def complete(self, *, system: str, user: str, schema: JSON) -> tuple[Any, JSON]:
        """(payload, raw). `payload` is a dict or a JSON string."""


class AnthropicClient:
    """Structured outputs (`output_config.format`) rather than a forced tool
    call: current models reject forced `tool_choice`. No refusal fallback is
    configured, because a silent swap to another model would contaminate an
    evaluation of this one; a refusal is recorded as a seat strike instead."""

    def __init__(self, spec: ModelSpec):
        import anthropic
        self.spec = spec
        self._client = anthropic.AsyncAnthropic(
            api_key=os.environ.get(spec.api_key_env or "ANTHROPIC_API_KEY"))

    async def complete(self, *, system: str, user: str, schema: JSON) -> tuple[Any, JSON]:
        output_config: dict[str, Any] = {"format": {"type": "json_schema", "schema": schema}}
        if self.spec.effort is not None:
            output_config["effort"] = self.spec.effort
        kwargs: dict[str, Any] = {
            "model": self.spec.model, "max_tokens": self.spec.max_tokens, "system": system,
            "messages": [{"role": "user", "content": user}], "output_config": output_config,
        }
        if self.spec.temperature is not None:
            kwargs["temperature"] = self.spec.temperature
        kwargs.update(self.spec.provider_params or {})
        resp = await self._client.messages.create(**kwargs)
        if resp.stop_reason == "refusal":
            details = getattr(resp, "stop_details", None)
            raise ProviderRefusal(getattr(details, "category", None) or "refusal")
        payload = next((b.text for b in resp.content if b.type == "text"), None)
        thinking = [b.thinking for b in resp.content if b.type == "thinking"]
        return payload, {
            "served_model": resp.model, "stop_reason": resp.stop_reason,
            "usage": {"input_tokens": resp.usage.input_tokens,
                      "output_tokens": resp.usage.output_tokens},
            "provider_reasoning": "\n".join(thinking) or None,
        }


class OpenAIClient:
    """OpenAI chat completions; with `openai_compat`, any compatible gateway
    (OpenRouter, vLLM, NIM) via `base_url`."""

    def __init__(self, spec: ModelSpec):
        from openai import AsyncOpenAI
        self.spec = spec
        default_env = "OPENAI_API_KEY" if spec.provider == "openai" else "OPENROUTER_API_KEY"
        self._client = AsyncOpenAI(base_url=spec.base_url,
                                   api_key=os.environ.get(spec.api_key_env or default_env))

    async def complete(self, *, system: str, user: str, schema: JSON) -> tuple[Any, JSON]:
        tokens_key = "max_completion_tokens" if self.spec.provider == "openai" else "max_tokens"
        kwargs: dict[str, Any] = {
            "model": self.spec.model, tokens_key: self.spec.max_tokens,
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": user}],
            "response_format": {"type": "json_schema",
                                "json_schema": {"name": "turn", "schema": schema}},
        }
        if self.spec.temperature is not None:
            kwargs["temperature"] = self.spec.temperature
        kwargs.update(self.spec.provider_params or {})
        resp = await self._client.chat.completions.create(**kwargs)
        msg = resp.choices[0].message
        usage = resp.usage
        return msg.content, {
            "served_model": resp.model, "stop_reason": resp.choices[0].finish_reason,
            "usage": {"input_tokens": getattr(usage, "prompt_tokens", 0) or 0,
                      "output_tokens": getattr(usage, "completion_tokens", 0) or 0},
            "provider_reasoning": getattr(msg, "reasoning", None)
            or getattr(msg, "reasoning_content", None),
        }


def build_client(spec: ModelSpec) -> ProviderClient:
    if spec.provider == "anthropic":
        return AnthropicClient(spec)
    if spec.provider in ("openai", "openai_compat"):
        return OpenAIClient(spec)
    raise ValueError(f"provider {spec.provider!r} has no API client")


def _usage_raw(spec: ModelSpec, text: TextInterface, attempts: list[JSON]) -> JSON:
    usage = {k: sum(a.get("usage", {}).get(k, 0) for a in attempts)
             for k in ("input_tokens", "output_tokens")}
    return {"model": spec.name, "model_id": spec.model, "prompt_version": text.prompt_version,
            "usage": usage, "attempts": attempts}


class ScriptedSeat:
    """A lineup entry played by a scripted policy, routed through render and
    parse so the full model path runs offline. Not adaptive."""

    def __init__(self, spec: ModelSpec, text: TextInterface, policy: Policy):
        self.spec, self.text, self.policy = spec, text, policy

    async def decide(self, req: DecisionRequest) -> DecisionResult:
        self.text.render(req.observation)
        payload = json.dumps({"reasoning": f"scripted:{self.policy.name}",
                              **self.policy.decide(req.observation)})
        action, reasoning = self.text.parse(payload)
        attempts = [{"served_model": f"scripted:{self.policy.name}", "payload": payload}]
        return DecisionResult(req.request_id, action, reasoning,
                              raw=_usage_raw(self.spec, self.text, attempts))


class ModelSeat:
    def __init__(self, spec: ModelSpec, text: TextInterface, client: ProviderClient):
        self.spec = spec
        self.text = text
        self.client = client

    async def decide(self, req: DecisionRequest) -> DecisionResult:
        system, user = self.text.render(req.observation)
        attempts: list[JSON] = []
        error: str | None = None
        for _ in range(2):
            if error is not None:
                user = (f"{user}\n\nYour previous reply was invalid ({error}). Reply again "
                        "with only a JSON object matching the schema.")
            started = time.monotonic()
            try:
                payload, raw = await asyncio.wait_for(
                    self.client.complete(system=system, user=user,
                                         schema=self.text.action_schema),
                    timeout=self.spec.timeout_s)
            except asyncio.TimeoutError:
                return self._result(req, attempts, strike="timeout")
            except ProviderRefusal as exc:
                return self._result(req, attempts, strike="refusal", detail=str(exc))
            except Exception as exc:  # provider/network errors are a seat failure, not a crash
                return self._result(req, attempts, strike="exception", detail=repr(exc))
            attempts.append({**raw, "payload": payload,
                             "latency_s": round(time.monotonic() - started, 3)})
            try:
                action, reasoning = self.text.parse(payload)
            except MalformedOutput as exc:
                error = str(exc)
                continue
            return self._result(req, attempts, action=action, reasoning=reasoning)
        return self._result(req, attempts, strike="malformed_output", detail=error)

    def _result(self, req: DecisionRequest, attempts: list[JSON], *, action: JSON | None = None,
                reasoning: str | None = None, strike: str | None = None,
                detail: str | None = None) -> DecisionResult:
        return DecisionResult(req.request_id, action, reasoning, strike, detail,
                              _usage_raw(self.spec, self.text, attempts))
