"""Harbor export (spike): parley-sim episodes as Harbor tasks.

Each Harbor task plays one episode. A `parley-runtime` sidecar holds the
episode and every counterpart seat; the agent under test plays the focal seat
through two MCP tools, `start_negotiation` and `take_turn`. The counterpart's
private values, and the episode itself, live only in the sidecar.

The verifier fetches the scored result from the sidecar's `/result` route with
a per-task token that only the sidecar and the verifier hold, so the agent
cannot write what gets scored. `/result` replays the trace (`verify`) before
reporting it.

Reward (`reward.json`):
  reward     focal_surplus_share as in local runs (can be negative: a deal
             below the agent's walk-away); 0 when the episode didn't finish
             or the share is undefined
  no_deal    1 when the episode ended without a deal
  completed  1 when the episode reached its end
  verified   1 when the trace replays exactly
"""

from __future__ import annotations

import asyncio
import hmac
import json
import os
import secrets
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from parley_sim import envs
from parley_sim.adapters.openenv import ParleySimAction, ParleySimEnvironment, ParleySimObservation
from parley_sim.runtime.episode import verify

CLIENT = "client"
RUNTIME_SERVICE = "parley-runtime"
RUNTIME_PORT = 8000

TOOL_START = "start_negotiation"
TOOL_TURN = "take_turn"


# --------------------------------------------------------------------------- runtime


@dataclass
class RuntimeConfig:
    """What one sidecar plays, read from its environment variables."""

    seed: int
    counterpart: str  # "policy:<name>" or "model:<lineup name>"
    token: str
    env: str = "support_desk"
    counterpart_seat: str = "customer"
    arm: str | None = None
    lineup: list[dict[str, Any]] | None = None
    seed_secret: str | None = None

    @classmethod
    def from_env(cls, environ: dict[str, str] | None = None) -> RuntimeConfig:
        e = os.environ if environ is None else environ
        token = e.get("PARLEY_RESULT_TOKEN", "")
        if len(token) < 16:
            raise ValueError("PARLEY_RESULT_TOKEN must be set (16+ characters)")
        lineup = json.loads(e["PARLEY_LINEUP"]) if e.get("PARLEY_LINEUP") else None
        return cls(seed=int(e["PARLEY_SEED"]), counterpart=e["PARLEY_COUNTERPART"], token=token,
                   env=e.get("PARLEY_ENV", "support_desk"),
                   counterpart_seat=e.get("PARLEY_COUNTERPART_SEAT", "customer"),
                   arm=e.get("PARLEY_ARM") or None, lineup=lineup,
                   seed_secret=e.get("PARLEY_SEED_SECRET") or None)


class HarborRuntime:
    """One episode, played by the agent through MCP. Calls are serialised:
    the episode has one pending focal decision at a time."""

    def __init__(self, config: RuntimeConfig, environment: ParleySimEnvironment | None = None):
        self.config = config
        self.entry = envs.get(config.env)
        self.environment = environment or ParleySimEnvironment(
            env=config.env, seats={self.entry.focal_seat: CLIENT,
                                   config.counterpart_seat: config.counterpart},
            lineup=config.lineup, seed_secret=config.seed_secret, release_trace="after_done")
        self._lock = asyncio.Lock()
        self._obs: ParleySimObservation | None = None

    def _describe(self, obs: ParleySimObservation) -> str:
        if obs.done:
            score = obs.outcomes.get(self.entry.focal_seat)
            return ("The negotiation is over."
                    + (f" Your score (value of the outcome to you): {score:g}." if score is not None else "")
                    + " Do not call any more tools.")
        note = (f"Your previous turn could not be used ({obs.last_strike}); "
                "a no-move turn was recorded instead.\n\n" if obs.last_strike else "")
        return f"{note}{obs.user_prompt}"

    async def start(self) -> str:
        async with self._lock:
            if self._obs is None:
                self._obs = await self.environment.reset_async(seed=self.config.seed,
                                                               arm=self.config.arm)
                return f"{self._obs.system_prompt}\n\n{self._describe(self._obs)}"
            return ("The negotiation has already started. Current state:\n\n"
                    + self._describe(self._obs))

    async def take_turn(self, message: str, move: dict[str, Any], explanation: str = "") -> str:
        async with self._lock:
            if self._obs is None:
                return f"Call {TOOL_START} first."
            if self._obs.done:
                return self._describe(self._obs)
            text = json.dumps({"message": message, "move": move, "explanation": explanation})
            try:
                action = ParleySimAction(text=text)
            except ValueError as exc:  # too long: same strike path as a malformed reply
                action = ParleySimAction(text=json.dumps({"error": str(exc)[:200]}))
            self._obs = await self.environment.step_async(action)
            return self._describe(self._obs)

    def result(self) -> dict[str, Any]:
        """The scored outcome, for the verifier only."""
        state = self.environment.state
        lines = state.trace
        completed = bool(state.done and lines and lines[-1].get("type") == "footer"
                         and lines[-1].get("status") == "complete")
        out: dict[str, Any] = {"completed": int(completed), "reward": 0.0, "no_deal": 0,
                               "verified": 0, "problems": [], "metrics": {}, "trace": lines}
        if not completed:
            out["problems"] = ["the episode did not finish"]
            return out
        problems = verify(self.entry.env(), lines, make_policy=self.entry.make_policy)
        metrics = self.entry.episode_metrics(lines)
        share = metrics.get("focal_surplus_share")
        out.update(verified=int(not problems), problems=problems, metrics=metrics,
                   reward=float(share) if share is not None and not problems else 0.0,
                   no_deal=int(metrics.get("deal") == 0.0))
        return out


def create_runtime_app(runtime: HarborRuntime) -> Any:
    """The sidecar's ASGI app: MCP at /mcp, /health, and token-guarded /result."""
    from mcp.server.fastmcp import FastMCP
    from mcp.server.transport_security import TransportSecuritySettings
    from starlette.requests import Request
    from starlette.responses import JSONResponse, PlainTextResponse

    schema = runtime.environment.text.action_schema
    move_schema = schema["properties"]["move"]
    mcp = FastMCP("parley-runtime", stateless_http=True, json_response=True,
                  transport_security=TransportSecuritySettings(enable_dns_rebinding_protection=False))

    @mcp.tool(name=TOOL_START, description=(
        "Start the negotiation (or see where it stands). Returns your role, your private "
        "information and your counterpart's opening turn."))
    async def start_negotiation() -> str:
        return await runtime.start()

    @mcp.tool(name=TOOL_TURN, description=(
        "Take your turn: a message to your counterpart and one move. `move` is "
        + json.dumps(move_schema) + ". Returns your counterpart's reply, or the end of the negotiation."))
    async def take_turn(message: str, move: dict[str, Any], explanation: str = "") -> str:
        return await runtime.take_turn(message, move, explanation)

    @mcp.custom_route("/health", methods=["GET"])
    async def health(request: Request) -> PlainTextResponse:
        return PlainTextResponse("ok")

    @mcp.custom_route("/result", methods=["GET"])
    async def result(request: Request) -> JSONResponse:
        given = request.headers.get("x-parley-token", "")
        if not hmac.compare_digest(given.encode(), runtime.config.token.encode()):
            return JSONResponse({"detail": "forbidden"}, status_code=403)
        return JSONResponse(runtime.result())

    return mcp.streamable_http_app()


def serve(host: str = "0.0.0.0", port: int = RUNTIME_PORT) -> None:
    import uvicorn

    uvicorn.run(create_runtime_app(HarborRuntime(RuntimeConfig.from_env())), host=host, port=port)


# --------------------------------------------------------------------------- export

INSTRUCTION = f"""\
You are taking part in a negotiation. You play one side; your counterpart plays the other.

Use the MCP tools on the `{RUNTIME_SERVICE}` server:
1. Call `{TOOL_START}` once. It tells you your role, your private information,
   the rules and your counterpart's first turn.
2. Then call `{TOOL_TURN}` once per turn with a `message`, a `move` and a short
   `explanation`, until it says the negotiation is over.

Your score is the value of the final outcome to you, as described by `{TOOL_START}`.
There are no files to write; the negotiation itself is graded.
"""

TASK_TOML = """\
schema_version = "1.1"

[task]
name = "{name}"
authors = [{{ name = "Parley Labs" }}]
keywords = ["parley-sim", "negotiation", "{env}", "{arm}"]

[metadata]
category = "negotiation"
seed = {seed}
counterpart = "{counterpart}"
arm = "{arm}"

[verifier]
timeout_sec = 120.0
env = {{ PARLEY_RESULT_TOKEN = "{token}" }}

[agent]
timeout_sec = {agent_timeout}

[environment]
build_timeout_sec = 900.0
cpus = 1
memory_mb = 2048

[[environment.mcp_servers]]
name = "{service}"
transport = "streamable-http"
url = "http://{service}:{port}/mcp"
"""

COMPOSE = """\
services:
  main:
    depends_on:
      {service}:
        condition: service_healthy
  {service}:
{source}
    environment:
      - PARLEY_ENV={env}
      - PARLEY_SEED={seed}
      - PARLEY_COUNTERPART={counterpart}
      - PARLEY_ARM={arm}
      - PARLEY_RESULT_TOKEN={token}
{extra_env}    expose:
      - "{port}"
    healthcheck:
      test: ["CMD", "python3", "-c",
             "import urllib.request; urllib.request.urlopen('http://localhost:{port}/health', timeout=2)"]
      interval: 2s
      timeout: 5s
      retries: 30
      start_period: 5s
"""

MAIN_DOCKERFILE = """\
FROM python:3.12-slim
{ca}WORKDIR /app
"""

RUNTIME_DOCKERFILE = """\
FROM python:3.12-slim
{ca}COPY dist/ /tmp/dist/
RUN pip install --no-cache-dir "$(ls /tmp/dist/*.whl)[openenv,openai]" "mcp>=1.30,<2" "uvicorn>=0.30" \\
    && rm -rf /tmp/dist
EXPOSE {port}
CMD ["python3", "-c", "from parley_sim.adapters.harbor import serve; serve()"]
"""

TEST_SH = """\
#!/bin/bash
set -euo pipefail
mkdir -p /logs/verifier
python3 /tests/fetch_result.py
"""

FETCH_RESULT = f"""\
import json, os, urllib.request

req = urllib.request.Request("http://{RUNTIME_SERVICE}:{RUNTIME_PORT}/result",
                             headers={{"x-parley-token": os.environ["PARLEY_RESULT_TOKEN"]}})
with urllib.request.urlopen(req, timeout=60) as resp:
    result = json.load(resp)
with open("/logs/verifier/trace.jsonl", "w") as f:
    for line in result.pop("trace"):
        f.write(json.dumps(line) + "\\n")
with open("/logs/verifier/result.json", "w") as f:
    json.dump(result, f, indent=2)
reward = {{k: result[k] for k in ("reward", "no_deal", "completed", "verified")}}
with open("/logs/verifier/reward.json", "w") as f:
    json.dump(reward, f)
print(json.dumps(reward))
if result["problems"]:
    print("problems:", result["problems"])
"""

# Plumbing check, not a strong player: hold one package, then accept the
# counterpart's standing proposal on the last turn (a deal is still refused if
# it's outside the agent's limits, which the trace records as a strike).
SOLVE_SH = f"""\
#!/bin/bash
set -euo pipefail
python3 - <<'PY'
import json, urllib.request

URL = "http://{RUNTIME_SERVICE}:{RUNTIME_PORT}/mcp"
HEAD = {{"content-type": "application/json", "accept": "application/json, text/event-stream"}}

def call(method, params, i=[0]):
    i[0] += 1
    body = json.dumps({{"jsonrpc": "2.0", "id": i[0], "method": method, "params": params}}).encode()
    with urllib.request.urlopen(urllib.request.Request(URL, body, HEAD), timeout=600) as r:
        return json.load(r)

def tool(name, **args):
    out = call("tools/call", {{"name": name, "arguments": args}})["result"]
    return "".join(c.get("text", "") for c in out["content"])

call("initialize", {{"protocolVersion": "2025-06-18", "capabilities": {{}},
                    "clientInfo": {{"name": "oracle", "version": "0"}}}})
text = tool("{TOOL_START}")
print(text)
for _ in range(20):
    if "negotiation is over" in text:
        break
    last = "Turns you have left, including this one: 1" in text
    move = ({{"type": "accept"}} if last else
            {{"type": "propose", "deal": {{"discount_pct": 20, "credit_usd": 25,
                                         "upgrade": "none", "term": "annual"}}}})
    text = tool("{TOOL_TURN}", message="Here is what I can do.", move=move,
                explanation="Plumbing check.")
    print(text)
PY
"""


def _arm(counterpart: str) -> str:
    kind, _, name = counterpart.partition(":")
    return f"scripted-{name}" if kind == "policy" else name


def export_task(out: Path, *, seed: int, counterpart: str, wheel: Path, env: str = "support_desk",
                dataset: str = "parley-labs/parley-sim", agent_timeout: float = 1800.0,
                lineup: list[dict[str, Any]] | None = None, extra_ca: Path | None = None) -> Path:
    """Write one Harbor task directory and return it. `extra_ca` adds a CA
    certificate to both images (for building behind a TLS-intercepting proxy);
    leave it unset for tasks you publish."""
    arm = _arm(counterpart)
    name = f"{env.replace('_', '-')}__s{seed}-{arm}"
    task = out / name
    if task.exists():
        shutil.rmtree(task)
    (task / "environment" / "runtime" / "dist").mkdir(parents=True)
    (task / "tests").mkdir()
    (task / "solution").mkdir()
    token = secrets.token_urlsafe(24)
    ca = ""
    if extra_ca is not None:
        for d in (task / "environment", task / "environment" / "runtime"):
            shutil.copy(extra_ca, d / "extra-ca.crt")
        ca = ("COPY extra-ca.crt /usr/local/share/ca-certificates/extra-ca.crt\n"
              "RUN cat /usr/local/share/ca-certificates/extra-ca.crt >> /etc/ssl/certs/ca-certificates.crt\n"
              "ENV SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt "
              "REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt "
              "NODE_EXTRA_CA_CERTS=/usr/local/share/ca-certificates/extra-ca.crt\n")
    shutil.copy(wheel, task / "environment" / "runtime" / "dist" / wheel.name)
    extra_env = (f"      - PARLEY_LINEUP={json.dumps(lineup)}\n" if lineup else "")
    fmt = dict(name=f"{dataset}__{name}", env=env, seed=seed, counterpart=counterpart, arm=arm,
               token=token, agent_timeout=agent_timeout, service=RUNTIME_SERVICE, port=RUNTIME_PORT,
               source="    build:\n      context: ./runtime", extra_env=extra_env, ca=ca)
    (task / "instruction.md").write_text(INSTRUCTION)
    (task / "task.toml").write_text(TASK_TOML.format(**fmt))
    (task / "environment" / "docker-compose.yaml").write_text(COMPOSE.format(**fmt))
    (task / "environment" / "Dockerfile").write_text(MAIN_DOCKERFILE.format(**fmt))
    (task / "environment" / "runtime" / "Dockerfile").write_text(RUNTIME_DOCKERFILE.format(**fmt))
    (task / "tests" / "test.sh").write_text(TEST_SH)
    (task / "tests" / "fetch_result.py").write_text(FETCH_RESULT)
    (task / "solution" / "solve.sh").write_text(SOLVE_SH)
    for sh in (task / "tests" / "test.sh", task / "solution" / "solve.sh"):
        sh.chmod(0o755)
    return task
