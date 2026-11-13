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

# "harness": calls are made by an external harness client (e.g. a NeMo Gym agent server).
Provider = Literal["anthropic", "openai", "openai_compat", "scripted", "harness"]


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
    # Condition this seat on a persona; "usersim" = NeMo UserSim's persona and
    # behavior model (see drivers/usersim_persona.py).
    persona: Literal["usersim"] | None = None
    # Play the seat as one growing conversation: each turn appends what changed
    # since the seat last acted, after its own earlier replies, instead of a
    # freshly rendered prompt. Training on NeMo Gym needs this, because one
    # rollout's token sequence must be continuous.
    conversation: bool = False

    @model_validator(mode="after")
    def _anthropic_has_no_sampling(self) -> ModelSpec:
        if self.provider == "anthropic" and self.temperature is not None:
            raise ValueError("the Anthropic API no longer accepts temperature; remove it")
        return self


class EpisodeAbort(Exception):
    """Infrastructure failure the seat must not absorb as a strike; the
    driver fails the episode instead (and may retry it)."""

    def __init__(self, reason: str, retryable: bool):
        super().__init__(reason)
        self.retryable = retryable


class ProviderRefusal(Exception):
    """The provider declined to answer (a safety stop, not malformed output).
    `raw` is the response metadata (served model, usage), so a refused call's
    tokens and cost are still counted."""

    def __init__(self, detail: str = "refusal", raw: JSON | None = None):
        super().__init__(detail)
        self.raw: JSON = raw or {}


class ProviderClient(Protocol):
    async def complete(self, *, system: str, user: str, schema: JSON,
                       history: list[JSON] | None = None) -> tuple[Any, JSON]:
        """(payload, raw). `payload` is a dict or a JSON string. `history`
        (earlier {"user", "assistant"} exchanges) is passed only to seats with
        `conversation: true`; a client used for them must accept it."""


def _history_messages(history: list[JSON] | None) -> list[JSON]:
    """Earlier exchanges of a conversation seat as chat messages."""
    out: list[JSON] = []
    for turn in history or []:
        out += [{"role": "user", "content": turn["user"]},
                {"role": "assistant", "content": turn["assistant"]}]
    return out


def _check_endpoint(name: str, url: str | None) -> None:
    """Keys travel only over TLS, except to a server on this machine."""
    if not url:
        return
    from urllib.parse import urlparse
    u = urlparse(url)
    if u.scheme == "https":
        return
    if u.scheme == "http" and u.hostname in ("localhost", "127.0.0.1", "::1"):
        return
    raise ValueError(f"{name}: base_url {url!r} must use https (plain http only to localhost)")


class AnthropicClient:
    """Structured outputs (`output_config.format`) rather than a forced tool
    call: current models reject forced `tool_choice`. No refusal fallback is
    configured, because a silent swap to another model would contaminate an
    evaluation of this one; a refusal is recorded as a seat strike instead."""

    def __init__(self, spec: ModelSpec):
        import anthropic
        self.spec = spec
        # The SDK reads ANTHROPIC_BASE_URL when no base_url is given, so guard
        # the endpoint the client will actually use.
        base_url = spec.base_url or os.environ.get("ANTHROPIC_BASE_URL") or None
        if base_url and not spec.api_key_env:
            raise ValueError(f"{spec.name}: a custom base_url needs api_key_env, so "
                             "ANTHROPIC_API_KEY is never sent to another host")
        _check_endpoint(spec.name, base_url)
        env = spec.api_key_env or "ANTHROPIC_API_KEY"
        key = os.environ.get(env)
        if not key:
            # Fail at startup, not on every call. A named variable never falls
            # back to ANTHROPIC_API_KEY. Behind a proxy that injects the real
            # key, any placeholder value works.
            raise RuntimeError(f"{env} is not set")
        self._client = anthropic.AsyncAnthropic(api_key=key, base_url=base_url)

    async def complete(self, *, system: str, user: str, schema: JSON,
                       history: list[JSON] | None = None) -> tuple[Any, JSON]:
        output_config: dict[str, Any] = {"format": {"type": "json_schema", "schema": schema}}
        if self.spec.effort is not None:
            output_config["effort"] = self.spec.effort
        kwargs: dict[str, Any] = {
            "model": self.spec.model, "max_tokens": self.spec.max_tokens, "system": system,
            "messages": [*_history_messages(history), {"role": "user", "content": user}],
            "output_config": output_config,
            "thinking": {"type": "adaptive", "display": "summarized"},
        }
        if self.spec.temperature is not None:
            kwargs["temperature"] = self.spec.temperature
        kwargs.update(self.spec.provider_params or {})
        resp = await self._client.messages.create(**kwargs)
        raw: JSON = {"served_model": resp.model, "stop_reason": resp.stop_reason,
                     "usage": {"input_tokens": resp.usage.input_tokens,
                               "output_tokens": resp.usage.output_tokens}}
        if resp.stop_reason == "refusal":
            details = getattr(resp, "stop_details", None)
            raise ProviderRefusal(getattr(details, "category", None) or "refusal", raw=raw)
        payload = next((b.text for b in resp.content if b.type == "text"), None)
        thinking = [b.thinking for b in resp.content if b.type == "thinking"]
        return payload, {**raw, "provider_reasoning": "\n".join(thinking) or None}


class OpenAIClient:
    """OpenAI chat completions; with `openai_compat`, any compatible gateway
    (OpenRouter, vLLM, NIM) via `base_url`."""

    def __init__(self, spec: ModelSpec):
        from openai import AsyncOpenAI
        self.spec = spec
        # A key must only ever reach the host it belongs to: an OpenAI key goes
        # to a custom base_url only when named explicitly, and a gateway key
        # never goes to OpenAI's default endpoint.
        # The OpenAI SDK reads OPENAI_BASE_URL when no base_url is given, so
        # guard the endpoint the client will actually use.
        base_url = spec.base_url or (os.environ.get("OPENAI_BASE_URL") or None
                                     if spec.provider == "openai" else None)
        if spec.provider == "openai" and base_url and not spec.api_key_env:
            raise ValueError(f"{spec.name}: provider 'openai' with a custom base_url needs "
                             "api_key_env, so OPENAI_API_KEY is never sent to another host")
        _check_endpoint(spec.name, base_url)
        if spec.provider == "openai_compat" and not spec.base_url:
            raise ValueError(f"{spec.name}: provider 'openai_compat' needs base_url")
        from urllib.parse import urlparse
        host = urlparse(base_url).hostname if base_url else None
        if spec.provider == "openai_compat" and not spec.api_key_env and host != "openrouter.ai":
            raise ValueError(f"{spec.name}: provider 'openai_compat' with base_url host {host!r} "
                             "needs api_key_env; OPENROUTER_API_KEY is sent only to openrouter.ai")
        env = spec.api_key_env or ("OPENAI_API_KEY" if spec.provider == "openai"
                                   else "OPENROUTER_API_KEY")
        key = os.environ.get(env)
        if not key:
            # Never let the SDK fall back to OPENAI_API_KEY: that would send an
            # OpenAI secret to whatever gateway base_url points at. Behind a
            # proxy that injects the real key, any placeholder value works.
            raise RuntimeError(f"{env} is not set")
        headers: dict[str, Any] | None = None
        if base_url and host != "api.openai.com":
            # The SDK fills OpenAI-Organization and OpenAI-Project from
            # OPENAI_ORG_ID / OPENAI_PROJECT_ID even when passed None; Omit()
            # removes them, so OpenAI account identifiers stay with OpenAI.
            from openai import Omit
            headers = {"OpenAI-Organization": Omit(), "OpenAI-Project": Omit()}
        self._client = AsyncOpenAI(base_url=base_url, api_key=key, default_headers=headers)

    async def complete(self, *, system: str, user: str, schema: JSON,
                       history: list[JSON] | None = None) -> tuple[Any, JSON]:
        compat = self.spec.provider != "openai"
        kwargs: dict[str, Any] = {
            "model": self.spec.model,
            ("max_tokens" if compat else "max_completion_tokens"): self.spec.max_tokens,
            "messages": [{"role": "system", "content": system}, *_history_messages(history),
                         {"role": "user", "content": user}],
            "response_format": {"type": "json_schema",
                                "json_schema": {"name": "turn", "schema": schema}},
        }
        if self.spec.temperature is not None:
            kwargs["temperature"] = self.spec.temperature
        params = dict(self.spec.provider_params or {})
        if self.spec.effort is not None:
            if compat:  # OpenRouter's unified reasoning control
                extra = dict(params.get("extra_body") or {})
                extra["reasoning"] = {**extra.get("reasoning", {}), "effort": self.spec.effort}
                params["extra_body"] = extra
            else:
                kwargs["reasoning_effort"] = self.spec.effort
        kwargs.update(params)
        resp = await self._client.chat.completions.create(**kwargs)
        usage = resp.usage
        cost = getattr(usage, "cost", None)  # OpenRouter, when usage accounting is on
        raw: JSON = {
            "served_model": resp.model, "stop_reason": resp.choices[0].finish_reason,
            "usage": {"input_tokens": getattr(usage, "prompt_tokens", 0) or 0,
                      "output_tokens": getattr(usage, "completion_tokens", 0) or 0,
                      **({"cost_usd": float(cost)} if cost is not None else {})},
        }
        if resp.choices[0].finish_reason == "content_filter":
            raise ProviderRefusal("content_filter", raw=raw)
        msg = resp.choices[0].message
        if getattr(msg, "refusal", None):  # structured-output refusals arrive with finish "stop"
            raise ProviderRefusal(f"refusal: {msg.refusal[:200]}", raw=raw)
        return msg.content, {**raw, "provider_reasoning": getattr(msg, "reasoning", None)
                             or getattr(msg, "reasoning_content", None)}


def build_client(spec: ModelSpec) -> ProviderClient:
    if spec.provider == "anthropic":
        return AnthropicClient(spec)
    if spec.provider in ("openai", "openai_compat"):
        return OpenAIClient(spec)
    raise ValueError(f"provider {spec.provider!r} has no API client")


def _usage_raw(spec: ModelSpec, text: TextInterface, attempts: list[JSON]) -> JSON:
    # Failed calls leave stub attempts with no usage; they count as attempts only.
    used = [a.get("usage") or {} for a in attempts]
    usage: JSON = {k: sum(u.get(k, 0) or 0 for u in used) for k in ("input_tokens", "output_tokens")}
    if any("cost_usd" in u for u in used):
        usage["cost_usd"] = sum(u.get("cost_usd", 0.0) or 0.0 for u in used)
    raw: JSON = {"model": spec.name, "model_id": spec.model, "prompt_version": text.prompt_version,
                 "usage": usage, "attempts": attempts}
    if getattr(text, "meta", None):
        raw["persona"] = text.meta
    return raw


class ProviderHalted(RuntimeError):
    """The run was stopped to protect the account (see Breaker)."""


class Breaker:
    """Shared by every seat in a run. Trips on any 401/402/403/404 or after
    `max_refusals` refusals; once tripped, seats raise instead of calling the
    provider again. Repeated refusals can get an account suspended, so a run
    that is being refused must stop, not grind through its grid."""

    def __init__(self, max_refusals: int = 3):
        self.max_refusals = max_refusals
        self.refusals = 0
        self.tripped: str | None = None

    def refusal(self, detail: str) -> None:
        self.refusals += 1
        if self.refusals >= self.max_refusals and self.tripped is None:
            self.tripped = f"{self.refusals} refusals (last: {detail})"

    def fatal(self, detail: str) -> None:
        self.tripped = self.tripped or detail

    def check(self) -> None:
        if self.tripped is not None:
            raise ProviderHalted(self.tripped)


# Bad key, out of credit, no access, unknown model: every later call would
# fail the same way, so the run halts.
_FATAL_STATUS = frozenset({401, 402, 403, 404})
# The request itself is invalid (bad parameter, unsupported option, prompt too
# long): the episode errors with the provider's message instead of being
# scored on fallback moves.
_REQUEST_ERROR_STATUS = frozenset({400, 422})
# Turns in a row whose calls all failed (timeouts, 429s, network errors)
# before a seat gives up and aborts the episode.
MAX_CONSECUTIVE_FAILURES = 3
# Stop reasons meaning the reply hit the output cap (Anthropic, OpenAI).
_TRUNCATED = frozenset({"max_tokens", "length"})


def _is_fatal(exc: Exception) -> bool:
    status = getattr(exc, "status_code", None)
    # Anthropic reports an exhausted credit balance as a 400, not a 402.
    return status in _FATAL_STATUS or (status == 400 and "credit balance" in str(exc).lower())


def _is_blank(payload: Any) -> bool:
    return payload is None or (isinstance(payload, str) and not payload.strip())


def _since(started: float) -> float:
    return round(time.monotonic() - started, 3)


class ScriptedSeat:
    """A lineup entry played by a scripted policy, routed through render and
    parse so the full model path runs offline. Not adaptive."""

    def __init__(self, spec: ModelSpec, text: TextInterface, policy: Policy):
        self.spec, self.text, self.policy = spec, text, policy

    async def decide(self, req: DecisionRequest) -> DecisionResult:
        self.text.render(req.observation)
        payload = json.dumps({**self.policy.decide(req.observation),
                              "explanation": f"scripted:{self.policy.name}"})
        action, explanation = self.text.parse(payload)
        attempts = [{"served_model": f"scripted:{self.policy.name}", "payload": payload}]
        return DecisionResult(req.request_id, action, raw=_usage_raw(self.spec, self.text, attempts),
                              explanation=explanation)


class ModelSeat:
    def __init__(self, spec: ModelSpec, text: TextInterface, client: ProviderClient,
                 breaker: Breaker | None = None):
        self.spec = spec
        self.text = text
        self.client = client
        self.breaker = breaker or Breaker()
        # Conversation mode: every completed exchange, in order, and the
        # observation at this seat's previous turn.
        self.history: list[JSON] = []
        self._previous: JSON | None = None
        # The action this seat's last turn submitted (None: no usable reply),
        # so the next update can say when the environment recorded otherwise.
        self._submitted: JSON | None = None
        # Turns in a row that ended in a timeout or exception strike.
        self._failures = 0

    def _prompt(self, observation: JSON) -> tuple[str, str]:
        system, user = self.text.render(observation)
        if self.spec.conversation and self._previous is not None:
            update = getattr(self.text, "render_update", None)
            if update is not None:
                user = update(self._previous, observation, self._submitted)
        return system, user

    async def decide(self, req: DecisionRequest) -> DecisionResult:
        system, user = self._prompt(req.observation)
        self._submitted = None
        attempts: list[JSON] = []
        error: str | None = None
        in_history = False  # whether the invalid reply being corrected is in the conversation
        for _ in range(2):
            self.breaker.check()
            if error is not None:
                correction = (f"Your previous reply was invalid ({error}). Reply again "
                              "with only a JSON object matching the schema.")
                # A conversation keeps an invalid reply and adds the correction as
                # the next turn; otherwise, or when there was no reply to keep,
                # the correction goes onto the same prompt.
                user = correction if in_history else f"{user}\n\n{correction}"
            started = time.monotonic()
            extra = {"history": list(self.history)} if self.spec.conversation else {}
            try:
                payload, raw = await asyncio.wait_for(
                    self.client.complete(system=system, user=user,
                                         schema=self.text.action_schema, **extra),
                    timeout=self.spec.timeout_s)
            except TimeoutError:
                detail = f"no reply within timeout_s={self.spec.timeout_s}"
                attempts.append({"error": detail, "latency_s": _since(started)})
                return self._failed(req, attempts, "timeout", detail)
            except EpisodeAbort:
                raise
            except ProviderRefusal as exc:
                self._failures = 0
                attempts.append({**exc.raw, "stop_reason": "refusal", "detail": str(exc),
                                 "latency_s": _since(started)})
                self.breaker.refusal(str(exc))
                return self._result(req, attempts, strike="refusal", detail=str(exc))
            except Exception as exc:  # provider/network errors are a seat failure, not a crash
                if _is_fatal(exc):
                    self.breaker.fatal(repr(exc))
                    raise ProviderHalted(repr(exc)) from exc
                if getattr(exc, "status_code", None) in _REQUEST_ERROR_STATUS:
                    raise EpisodeAbort(f"{self.spec.name} at seat {req.seat}: provider rejected "
                                       f"the request: {exc!r}"[:2000], retryable=False) from exc
                attempts.append({"error": repr(exc)[:2000], "latency_s": _since(started)})
                return self._failed(req, attempts, "exception", repr(exc))
            self._failures = 0
            attempts.append({**raw, "payload": payload, "latency_s": _since(started)})
            blank = _is_blank(payload)
            if self.spec.conversation and not blank:
                # An empty reply never enters the conversation (the Anthropic API
                # rejects empty assistant turns); it is retried like a failed call.
                self.history.append({"user": user, "assistant": payload if isinstance(payload, str)
                                     else json.dumps(payload)})
                # Only a turn that reached the conversation moves its start point:
                # after a refused or failed call, the next update still covers
                # everything since the last reply the model actually gave.
                self._previous = req.observation
            try:
                if blank:
                    raise MalformedOutput("empty reply")
                action, explanation = self.text.parse(payload)
            except MalformedOutput as exc:
                if raw.get("stop_reason") in _TRUNCATED:
                    # Cut off by the output cap, not malformed by the model; the
                    # same request would be cut off again, so no repair retry.
                    return self._result(req, attempts, strike="truncated", detail=(
                        f"reply cut off at max_tokens={self.spec.max_tokens} "
                        f"(stop_reason {raw.get('stop_reason')!r})"))
                error = str(exc)
                in_history = self.spec.conversation and not blank
                continue
            self._submitted = action
            return self._result(req, attempts, action=action,
                                reasoning=raw.get("provider_reasoning"), explanation=explanation)
        return self._result(req, attempts, strike="malformed_output", detail=error)

    def _failed(self, req: DecisionRequest, attempts: list[JSON], strike: str,
                detail: str) -> DecisionResult:
        """A turn whose call failed outright. A seat whose calls keep failing
        (persistent 429s, a dead network) aborts the episode rather than let it
        finish on fallback moves."""
        self._failures += 1
        if self._failures >= MAX_CONSECUTIVE_FAILURES:
            raise EpisodeAbort(f"{self.spec.name} at seat {req.seat}: {self._failures} turns in a "
                               f"row failed ({strike}); last error: {detail}"[:2000],
                               retryable=True)
        return self._result(req, attempts, strike=strike, detail=detail)

    def _result(self, req: DecisionRequest, attempts: list[JSON], *, action: JSON | None = None,
                reasoning: str | None = None, explanation: str | None = None,
                strike: str | None = None, detail: str | None = None) -> DecisionResult:
        return DecisionResult(req.request_id, action, reasoning, strike, detail,
                              _usage_raw(self.spec, self.text, attempts), explanation)
