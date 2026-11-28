"""Harbor export (experimental): parley-sim episodes as Harbor tasks.

`parley-sim harbor-export` turns a matched-run config into Harbor tasks: one
task per cell of the config's grid (seed, arm, replicate), with the same seeds, arms and
counterpart rotation as a local run. The agent under test takes the focal
seat's place; it is chosen when the tasks are run (`harbor run -a ... -m ...`),
so the config's `focal_models` are not used.

Each task's `parley-runtime` sidecar holds the episode and every counterpart
seat (scripted policy, model, or UserSim persona model). The agent plays the
focal seat through two MCP tools, `start_episode` and `take_turn`; a turn is
one action in the environment's action schema, checked as a model seat's reply
would be. The counterparts' private information, models and API keys live only
in the sidecar. Any built-in or installed environment works; the sidecar image
must contain it.

The verifier fetches the result from the sidecar's `/result` route with a
per-task token that only the sidecar and the verifier hold, so the agent cannot
write what gets scored. The sidecar replays the trace (`verify`) before
reporting it. Reward (`reward.json`):

  reward     the cell's reward metric, an episode metric of the environment
             (default focal_surplus_share, as in local runs); 0 if the episode
             didn't finish, didn't verify, or the metric is undefined
  no_deal    1 when the episode finished without a deal; only for
             environments with a `deal` metric
  completed  1 when the episode reached its end
  verified   1 when the trace replays exactly

`collect_jobs` turns finished Harbor trials back into a run directory that
`parley-sim report` reads, with each Harbor agent and model as a focal model.

Limits: the scenario of a seed is computable from the public generator. A
published task names its seed (in task files the agent's container doesn't
see), so an agent with internet access could in principle look it up; export
with a seed secret for private evaluations.
"""

from __future__ import annotations

import asyncio
import base64
import hmac
import json
import logging
import os
import re
import secrets
import shutil
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Annotated, Any

from parley_sim import envs
from parley_sim.core.hashing import digest
from parley_sim.core.protocol import JSON, MalformedOutput, Seat
from parley_sim.core.rng import secret_seed
from parley_sim.drivers.models import (
    Breaker,
    ModelSeat,
    ModelSpec,
    ProviderClient,
    ScriptedSeat,
    aclose_clients,
    build_client,
)
from parley_sim.runtime.episode import (
    DecisionRequest,
    DecisionResult,
    Episode,
    EpisodeComplete,
    EpisodeSpec,
    verify,
)
from parley_sim.trace import read_trace

_log = logging.getLogger(__name__)

CLIENT = "client"  # the focal seat's label: played from outside, as in the OpenEnv adapter
RUNTIME_SERVICE = "parley-runtime"
RUNTIME_PORT = 8000
PERSONA_PATH = "/data/personas.parquet"  # where the user's persona panel is mounted
TOOL_START = "start_episode"
TOOL_TURN = "take_turn"
MAX_ACTION_CHARS = 20_000
_PARSE_ERRORS = (MalformedOutput, RecursionError, ValueError, TypeError)
_JSON_REPLY = re.compile(r"\n*Respond with a JSON object matching this schema:\n.*\Z", re.S)


# --------------------------------------------------------------------------- cells


@dataclass
class HarborCell:
    """One task's episode: everything the sidecar needs except secrets."""

    run_id: str
    env: str
    seed: int
    arm: str
    replicate: int
    focal_seat: Seat
    seats: dict[Seat, str]  # counterpart seats only: "policy:<name>" | "model:<name>"
    env_config: dict[str, Any] = field(default_factory=dict)
    lineup: list[dict[str, Any]] = field(default_factory=list)  # specs of the model seats
    reward_metric: str = "focal_surplus_share"
    config_hash: str | None = None  # of the config the task was exported from; set by export_task

    @property
    def name(self) -> str:
        return f"{self.env.replace('_', '-')}__s{self.seed}-{self.arm}-r{self.replicate}"

    def encode(self) -> str:
        return base64.b64encode(json.dumps(asdict(self), sort_keys=True).encode()).decode()

    @classmethod
    def decode(cls, text: str) -> HarborCell:
        return cls(**json.loads(base64.b64decode(text)))


def harbor_cells(grid: list[Any], *, run_id: str, env: str, env_config: dict[str, Any],
                 lineup: list[ModelSpec], reward_metric: str = "focal_surplus_share") -> list[HarborCell]:
    """A matched grid without its focal model: one cell per (seed, arm,
    replicate), counterparts seated as a local run would seat them. `grid` is
    `parley_sim.matched.build_grid(cfg)`; the cells of its first focal model
    are used."""
    kinds = envs.get(env, allow_import=True).metric_kinds
    if reward_metric not in kinds:
        raise ValueError(f"reward metric {reward_metric!r} is not one of {sorted(kinds)}")
    specs = {m.name: m for m in lineup}
    focal = grid[0].focal if grid else None
    out = []
    for c in grid:
        if c.focal != focal:
            continue
        seats = {s: label for s, label in c.seats.items() if s != c.focal_seat}
        members = sorted({label.split(":", 1)[1] for label in seats.values()
                          if label.startswith("model:")})
        out.append(HarborCell(run_id=run_id, env=env, seed=c.seed, arm=c.arm,
                              replicate=c.replicate, focal_seat=c.focal_seat, seats=seats,
                              env_config=dict(env_config),
                              lineup=[specs[n].model_dump(exclude_none=True) for n in members],
                              reward_metric=reward_metric))
    return out


def key_env(spec: ModelSpec) -> str | None:
    """The environment variable a model seat reads its API key from (as
    `build_client` picks it)."""
    if spec.provider == "scripted":
        return None
    if spec.api_key_env:
        return spec.api_key_env
    return {"anthropic": "ANTHROPIC_API_KEY", "openai": "OPENAI_API_KEY"}.get(
        spec.provider, "OPENROUTER_API_KEY")


# --------------------------------------------------------------------------- runtime


@dataclass
class RuntimeConfig:
    """What one sidecar plays, read from its environment variables."""

    cell: HarborCell
    token: str
    seed_secret: str | None = None
    persona_file: str | None = None

    @classmethod
    def from_env(cls, environ: dict[str, str] | None = None) -> RuntimeConfig:
        e = os.environ if environ is None else environ
        token = e.get("PARLEY_RESULT_TOKEN", "")
        if len(token) < 16:
            raise ValueError("PARLEY_RESULT_TOKEN must be set (16+ characters)")
        return cls(cell=HarborCell.decode(e["PARLEY_CELL"]), token=token,
                   seed_secret=e.get("PARLEY_SEED_SECRET") or None,
                   persona_file=e.get("PARLEY_PERSONA_FILE") or None)


class HarborRuntime:
    """One episode, played by the agent through MCP. Calls are serialised:
    the episode has one pending focal decision at a time."""

    def __init__(self, config: RuntimeConfig, client_factory=build_client):
        self.config = config
        cell = config.cell
        # The environment the task was exported with: the operator's own config
        # (an import path like `pkg.envs:ENTRY` is imported, as `matched` does),
        # not a name read from a trace or result.
        self.entry = envs.get(cell.env, allow_import=True)
        self.text = self.entry.text()
        self.lineup = {m["name"]: ModelSpec(**m) for m in cell.lineup}
        self.personas: list[dict[str, Any]] | None = None
        if any(m.persona for m in self.lineup.values()):
            if not config.persona_file:
                raise ValueError("a persona counterpart needs PARLEY_PERSONA_FILE")
            from parley_sim.drivers.usersim_persona import load_personas
            self.personas = load_personas(config.persona_file)
        self.clients: dict[str, ProviderClient] = {}
        for spec in self.lineup.values():  # fail at startup, not at the agent's first call
            if spec.provider != "scripted":
                self.clients[spec.name] = client_factory(spec)
        self.breaker = Breaker(3)
        self.episode: Episode | None = None
        self.counterparts: dict[Seat, ModelSeat | ScriptedSeat] = {}
        self.pending: DecisionRequest | None = None
        self.last_strike: str | None = None
        self.error: str | None = None
        self._lock = asyncio.Lock()

    def _seat_text(self, spec: ModelSpec):
        if spec.persona == "usersim":
            from parley_sim.drivers.usersim_persona import UserSimPersonaText, persona_for_seed
            persona = persona_for_seed(self.personas, self.config.cell.seed)
            return UserSimPersonaText(self.entry.text(), persona)
        return self.entry.text()

    def _build(self) -> Episode:
        cell = self.config.cell
        seed = secret_seed(self.config.seed_secret, cell.seed) if self.config.seed_secret else cell.seed
        seats = {cell.focal_seat: CLIENT, **cell.seats}
        spec = EpisodeSpec(seed=seed, seats=seats, config=cell.env_config, focal_seat=cell.focal_seat,
                           arm=cell.arm, pair_key=f"{cell.run_id}-{cell.seed}",
                           cell_key=f"{cell.run_id}-{cell.seed}-{cell.arm}-r{cell.replicate}",
                           replicate=cell.replicate)
        policies = {s: self.entry.make_policy(label.split(":", 1)[1], s)
                    for s, label in cell.seats.items() if label.startswith("policy:")}
        for s, label in cell.seats.items():
            if label.startswith("model:"):
                m = self.lineup[label.split(":", 1)[1]]
                self.counterparts[s] = (
                    ScriptedSeat(m, self._seat_text(m), self.entry.make_policy(m.model, s))
                    if m.provider == "scripted"
                    else ModelSeat(m, self._seat_text(m), self.clients[m.name], self.breaker))
        return Episode(self.entry.env(), spec, policies)

    async def _advance(self, event: DecisionRequest | EpisodeComplete) -> None:
        """Play counterpart turns until the focal seat must act or the episode ends."""
        episode = self.episode
        assert episode is not None
        try:
            while isinstance(event, DecisionRequest) and event.seat != self.config.cell.focal_seat:
                event = episode.advance(await self.counterparts[event.seat].decide(event))
        except BaseException as exc:
            # Any interruption, including a cancelled tool call, ends the
            # episode as errored: the focal request it held has already been
            # consumed, so it must not stay pending.
            if episode.result is None:
                episode.abort(repr(exc))
            self.error = repr(exc)[:500]
            self.pending = None
            if not isinstance(exc, Exception):
                raise
            event = episode.result
        self.pending = event if isinstance(event, DecisionRequest) else None

    def _prompt(self) -> str:
        episode = self.episode
        if episode is None:
            return f"Call {TOOL_START} first."
        if self.pending is None:
            if self.error is not None:
                return ("The episode ended early: a counterpart could not continue. "
                        "Do not call any more tools.")
            score = (episode.result.outcomes or {}).get(self.config.cell.focal_seat) \
                if episode.result else None
            return ("The episode is over."
                    + (f" Your score (value of the outcome to you): {score:g}." if score is not None else "")
                    + " Do not call any more tools.")
        _, user = self.text.render(self.pending.observation)
        user = _JSON_REPLY.sub("", user) + f"\n\nTake your turn with `{TOOL_TURN}`."

        note = (f"Your previous turn could not be used ({self.last_strike}); "
                "no move was made for it.\n\n" if self.last_strike else "")
        return note + user

    async def start(self) -> str:
        async with self._lock:
            if self.episode is not None:
                return "The episode has already started. Current state:\n\n" + self._prompt()
            self.episode = self._build()
            await self._advance(self.episode.advance())
            if self.pending is None:
                return self._prompt()
            system, _ = self.text.render(self.pending.observation)
            return f"{system}\n\n{self._prompt()}"

    def _focal_result(self, req: DecisionRequest, submitted: JSON) -> DecisionResult:
        raw: JSON = {"source": "harbor", "payload": submitted}
        try:
            if len(json.dumps(submitted)) > MAX_ACTION_CHARS:
                raise MalformedOutput(f"turn is longer than {MAX_ACTION_CHARS} characters")
            parsed, explanation = self.text.parse(submitted)
        except _PARSE_ERRORS as exc:
            detail = str(exc) if isinstance(exc, MalformedOutput) else f"{type(exc).__name__}: {exc}"
            return DecisionResult(req.request_id, None, strike="malformed_output",
                                  detail=detail[:500], raw=raw)
        return DecisionResult(req.request_id, parsed, explanation=explanation, raw=raw)

    async def take_turn(self, action: Any) -> str:
        async with self._lock:
            episode, req = self.episode, self.pending
            if episode is None or req is None:
                return self._prompt()
            result = self._focal_result(req, action)
            event = episode.advance(result)
            self.pending = None  # consumed; _advance sets the next one
            decision = next(l for l in reversed(episode.lines) if l["type"] == "decision"
                            and l["seat"] == self.config.cell.focal_seat)
            self.last_strike = decision["strike"]
            await self._advance(event)
            return self._prompt()

    def result(self) -> dict[str, Any]:
        """The scored outcome, for the verifier only. Mid-episode the trace is
        the focal seat's own view: no counterpart decisions or private values."""
        cell = self.config.cell
        out: dict[str, Any] = {
            "cell": {"run_id": cell.run_id, "config_hash": cell.config_hash, "seed": cell.seed,
                     "arm": cell.arm, "replicate": cell.replicate, "seats": cell.seats},
            "persona_digest": digest(self.personas) if self.personas is not None else None,
            "completed": 0, "errored": 0, "reward": 0.0, "verified": 0,
            "problems": [], "metrics": {}, "trace": []}
        episode = self.episode
        if episode is None:
            out["problems"] = ["the episode was never started"]
            return out
        lines = episode.lines
        if episode.result is None:
            out["problems"] = ["the episode did not finish"]
            out["trace"] = [l if l["type"] == "event" else {k: v for k, v in l.items() if k != "context"}
                            for l in lines if l["type"] == "event"
                            or (l["type"] == "decision" and l["seat"] == cell.focal_seat)]
            return out
        out["trace"] = list(lines)
        if episode.result.status != "complete":
            out.update(errored=1, problems=[f"episode errored: {self.error or episode.result.error}"])
            return out
        problems = verify(self.entry.env(), lines, make_policy=self.entry.make_policy)
        metrics = self.entry.episode_metrics(lines)
        value = metrics.get(cell.reward_metric)
        out.update(completed=1, verified=int(not problems), problems=problems, metrics=metrics,
                   reward=float(value) if value is not None and not problems else 0.0)
        if "deal" in metrics:
            out["no_deal"] = int(metrics["deal"] == 0.0)
        return out

    async def aclose(self) -> None:
        await aclose_clients(list(self.clients.values()))


def inline_refs(schema: JSON) -> JSON:
    """`schema` with its local `$ref`s (to `#/$defs/...` or `#/definitions/...`)
    replaced by the definitions they name, and those tables dropped, so it can
    be embedded in another schema (an MCP tool's input). Raises ValueError for
    a recursive or unresolvable reference."""
    tables = {"#/$defs/": schema.get("$defs") or {}, "#/definitions/": schema.get("definitions") or {}}

    def resolve(node: Any, seen: frozenset[str]) -> Any:
        if isinstance(node, list):
            return [resolve(v, seen) for v in node]
        if not isinstance(node, dict):
            return node
        ref = node.get("$ref")
        if isinstance(ref, str):
            prefix = next((p for p in tables if ref.startswith(p)), None)
            name = ref[len(prefix):] if prefix else None
            if name is None or name not in tables[prefix]:
                raise ValueError(f"unresolvable $ref {ref!r}")
            if ref in seen:
                raise ValueError(f"recursive $ref {ref!r}")
            target = resolve(tables[prefix][name], seen | {ref})
            return {**target, **{k: resolve(v, seen) for k, v in node.items() if k != "$ref"}}
        return {k: resolve(v, seen) for k, v in node.items() if k not in ("$defs", "definitions")}

    return resolve(schema, frozenset())


def create_runtime_app(runtime: HarborRuntime) -> Any:
    """The sidecar's ASGI app: MCP at /mcp, /health, and token-guarded /result."""
    from mcp.server.fastmcp import FastMCP
    from mcp.server.transport_security import TransportSecuritySettings
    from pydantic import Field
    from starlette.requests import Request
    from starlette.responses import JSONResponse, PlainTextResponse

    schema = runtime.text.action_schema
    try:
        tool_schema: JSON | None = inline_refs(schema)
    except ValueError as exc:  # recursive: describe it in text, accept any object
        _log.warning("action schema can't be inlined into the tool schema: %s", exc)
        tool_schema = None
    mcp = FastMCP("parley-runtime", stateless_http=True, json_response=True,
                  transport_security=TransportSecuritySettings(enable_dns_rebinding_protection=False))

    @mcp.tool(name=TOOL_START, description=(
        "Start the episode (or see where it stands). Returns your role, the rules, your "
        "private information and what has happened so far."))
    async def start_episode() -> str:
        return await runtime.start()

    async def take_turn(action: dict[str, Any]) -> str:
        return await runtime.take_turn(action)

    # A real annotation object (this module defers annotations), so the tool's
    # input schema carries the environment's action schema.
    if tool_schema is not None:
        take_turn.__annotations__["action"] = Annotated[dict[str, Any], Field(json_schema_extra=tool_schema)]
    mcp.tool(name=TOOL_TURN, description=(
        "Take your turn: one action matching this JSON schema: " + json.dumps(tool_schema or schema)
        + ". Returns what happens next, or the end of the episode."))(take_turn)

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
You are taking part in an interactive episode with one or more counterparts.

Use the MCP tools on the `{RUNTIME_SERVICE}` server:
1. Call `{TOOL_START}` once. It tells you your role, the rules, your private
   information and what has happened so far.
2. Then call `{TOOL_TURN}` once per turn with one action, as its schema describes,
   until it says the episode is over.

How you are scored is described by `{TOOL_START}`. There are no files to write;
the episode itself is graded.
"""

TASK_TOML = """\
schema_version = "1.1"

[task]
name = "{task_name}"
authors = [{{ name = "Parley Labs" }}]
keywords = ["parley-sim", "simulated-users", "{env}", "{arm}"]

[metadata]
category = "interactive"
run_id = "{run_id}"
seed = {seed}
arm = "{arm}"
replicate = {replicate}
config_hash = "{config_hash}"

[verifier]
timeout_sec = 120.0
env = {{ PARLEY_RESULT_TOKEN = "{token}" }}

[agent]
timeout_sec = {agent_timeout}

[environment]
build_timeout_sec = 1200.0
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
      PARLEY_CELL: "{cell}"
      PARLEY_RESULT_TOKEN: "{token}"
{extra_env}{volumes}    expose:
      - "{port}"
    healthcheck:
      test: ["CMD", "python3", "-c",
             "import urllib.request; urllib.request.urlopen('http://localhost:{port}/health', timeout=2)"]
      interval: 2s
      timeout: 5s
      retries: 60
      start_period: 5s
"""

MAIN_DOCKERFILE = """\
FROM python:3.12-slim
{ca}WORKDIR /app
"""

RUNTIME_DOCKERFILE = """\
FROM python:3.12-slim
{ca}COPY dist/ /tmp/dist/
RUN pip install --no-cache-dir "$(ls /tmp/dist/*.whl)[{extras}]" && rm -rf /tmp/dist
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
reward = {{k: result[k] for k in ("reward", "no_deal", "completed", "verified") if k in result}}
with open("/logs/verifier/reward.json", "w") as f:
    json.dump(reward, f)
print(json.dumps(reward))
if result["problems"]:
    print("problems:", result["problems"])
"""

# Plumbing check, not a player: every turn submits the smallest action the
# schema allows (preferring a no-op move), so the episode runs to its end.
SOLVE_SH = f"""\
#!/bin/bash
set -euo pipefail
python3 - <<'PY'
import base64, json, urllib.request

URL = "http://{RUNTIME_SERVICE}:{RUNTIME_PORT}/mcp"
HEAD = {{"content-type": "application/json", "accept": "application/json, text/event-stream"}}
SCHEMA = json.loads(base64.b64decode("__SCHEMA__"))

def call(method, params, i=[0]):
    i[0] += 1
    body = json.dumps({{"jsonrpc": "2.0", "id": i[0], "method": method, "params": params}}).encode()
    with urllib.request.urlopen(urllib.request.Request(URL, body, HEAD), timeout=900) as r:
        return json.load(r)

def tool(name, **args):
    out = call("tools/call", {{"name": name, "arguments": args}})["result"]
    return "".join(c.get("text", "") for c in out["content"])

def minimal(s):
    if "const" in s:
        return s["const"]
    if "enum" in s:
        return next((x for x in ("none", "pass", "wait", "noop") if x in s["enum"]), s["enum"][0])
    for key in ("anyOf", "oneOf"):
        if key in s:
            return minimal(s[key][0])
    kind = s.get("type")
    if kind == "object":
        props = s.get("properties", {{}})
        return {{k: minimal(props.get(k, {{}})) for k in s.get("required", [])}}
    if kind in ("integer", "number"):
        return s.get("minimum", 0)
    if kind == "boolean":
        return False
    if kind == "array":
        return []
    return "Plumbing check."

call("initialize", {{"protocolVersion": "2025-06-18", "capabilities": {{}},
                    "clientInfo": {{"name": "oracle", "version": "0"}}}})
text = tool("{TOOL_START}")
print(text)
for _ in range(200):
    if "Do not call any more tools" in text:
        break
    text = tool("{TOOL_TURN}", action=minimal(SCHEMA))
    print(text)
PY
"""

_CA_LINES = ("COPY extra-ca.crt /usr/local/share/ca-certificates/extra-ca.crt\n"
             "RUN cat /usr/local/share/ca-certificates/extra-ca.crt >> /etc/ssl/certs/ca-certificates.crt\n"
             "ENV SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt "
             "REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt "
             "NODE_EXTRA_CA_CERTS=/usr/local/share/ca-certificates/extra-ca.crt\n")


def export_task(out: Path, cell: HarborCell, *, chash: str, wheel: Path | None = None,
                image: str | None = None, dataset: str = "parley-labs/parley-sim",
                agent_timeout: float = 1800.0, seed_secret: str | None = None,
                extra_ca: Path | None = None) -> Path:
    """Write one Harbor task directory and return it. The sidecar is either
    built from `wheel` or pulled as a prebuilt `image`. `extra_ca` adds a CA
    certificate to the images (for building behind a TLS-intercepting proxy);
    leave it unset for tasks you publish."""
    if (wheel is None) == (image is None):
        raise ValueError("give exactly one of wheel or image")
    cell = replace(cell, config_hash=chash)
    task = out / cell.name
    if task.exists():
        shutil.rmtree(task)
    (task / "environment").mkdir(parents=True)
    (task / "tests").mkdir()
    (task / "solution").mkdir()
    token = secrets.token_urlsafe(24)
    specs = [ModelSpec(**m) for m in cell.lineup]
    persona = any(m.persona for m in specs)
    extras = ["harbor", *sorted({"anthropic" if m.provider == "anthropic" else "openai"
                                 for m in specs if m.provider != "scripted"})]
    if persona:
        extras.append("usersim")
    ca = ""
    if extra_ca is not None:
        ca = _CA_LINES
        shutil.copy(extra_ca, task / "environment" / "extra-ca.crt")
    if wheel is not None:
        runtime = task / "environment" / "runtime"
        (runtime / "dist").mkdir(parents=True)
        shutil.copy(wheel, runtime / "dist" / wheel.name)
        (runtime / "Dockerfile").write_text(RUNTIME_DOCKERFILE.format(ca=ca, extras=",".join(extras),
                                                                     port=RUNTIME_PORT))
        if extra_ca is not None:
            shutil.copy(extra_ca, runtime / "extra-ca.crt")
        source = "    build:\n      context: ./runtime"
    else:
        source = f"    image: {image}"
    keys = sorted({k for m in specs if (k := key_env(m))})
    extra_env = "".join(f'      {k}: "${{{k}:-}}"\n' for k in keys)
    if seed_secret:
        extra_env += f'      PARLEY_SEED_SECRET: "{seed_secret}"\n'
    volumes = ""
    if persona:
        extra_env += f'      PARLEY_PERSONA_FILE: "{PERSONA_PATH}"\n'
        volumes = ("    volumes:\n      - ${PARLEY_PERSONA_PANEL:?set PARLEY_PERSONA_PANEL to your "
                   f"UserSim persona panel}}:{PERSONA_PATH}:ro\n")
    fmt = dict(task_name=f"{dataset}__{cell.name}", env=cell.env, run_id=cell.run_id, seed=cell.seed,
               arm=cell.arm, replicate=cell.replicate, config_hash=chash, token=token,
               agent_timeout=agent_timeout, service=RUNTIME_SERVICE, port=RUNTIME_PORT,
               source=source, cell=cell.encode(), extra_env=extra_env, volumes=volumes, ca=ca)
    (task / "instruction.md").write_text(INSTRUCTION)
    (task / "task.toml").write_text(TASK_TOML.format(**fmt))
    (task / "environment" / "docker-compose.yaml").write_text(COMPOSE.format(**fmt))
    (task / "environment" / "Dockerfile").write_text(MAIN_DOCKERFILE.format(**fmt))
    (task / "tests" / "test.sh").write_text(TEST_SH)
    (task / "tests" / "fetch_result.py").write_text(FETCH_RESULT)
    schema = envs.get(cell.env, allow_import=True).text().action_schema
    try:
        schema = inline_refs(schema)  # the plumbing check follows no $refs
    except ValueError:
        pass
    (task / "solution" / "solve.sh").write_text(
        SOLVE_SH.replace("__SCHEMA__", base64.b64encode(json.dumps(schema).encode()).decode()))
    for sh in (task / "tests" / "test.sh", task / "solution" / "solve.sh"):
        sh.chmod(0o755)
    return task


# --------------------------------------------------------------------------- collect


def _focal_name(agent: dict[str, Any]) -> str:
    name = agent.get("name") or agent.get("import_path") or "agent"
    model = agent.get("model_name")
    return re.sub(r"[^A-Za-z0-9._-]+", "-", f"{name}.{model}" if model else name)


def _usage(lines: list[dict[str, Any]]) -> dict[str, float]:
    total = {"input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0}
    for line in lines:
        usage = ((line.get("raw") or {}).get("usage") or {}) if line.get("type") == "decision" else {}
        for k in total:
            total[k] += usage.get(k, 0)
    return total


def collect_jobs(config: dict[str, Any], chash: str, job_dirs: list[Path], out: Path) -> dict[str, int]:
    """Gather finished Harbor trials of tasks exported from a matched config
    (`config`: its model_dump, `chash`: its config_hash) into a run directory
    for `parley-sim report`. Each Harbor agent and model becomes a focal model.
    Only trials of tasks exported from this exact config (same config hash)
    are used; then one trial per (task, agent) is kept, the earliest.
    Counterpart usage is counted; the agent's own usage is in Harbor's results."""
    run_id = config["run_id"]
    trials = sorted({p.parent.parent for d in job_dirs for p in Path(d).rglob("verifier/result.json")},
                    key=lambda t: (json.loads((t / "result.json").read_text()).get("started_at") or "")
                    if (t / "result.json").exists() else "")
    rows: dict[tuple[int, str, str, int], dict[str, Any]] = {}
    focals: dict[str, dict[str, Any]] = {}
    counts = {"trials": 0, "kept": 0, "duplicates": 0, "other_config": 0, "no_config_hash": 0}
    persona_digest = None
    copies: list[tuple[Path, str]] = []  # written only once every trial has been checked
    for trial in trials:
        counts["trials"] += 1
        result = json.loads((trial / "verifier" / "result.json").read_text())
        trial_config = json.loads((trial / "config.json").read_text())
        cell = result.get("cell") or {}
        # Checked before deduplication, so a trial of another config can never
        # stand in for (or crowd out) one of this config's.
        if not cell.get("config_hash"):
            counts["no_config_hash"] += 1  # exported before the hash was recorded
            continue
        if cell["config_hash"] != chash or cell.get("run_id") != run_id:
            counts["other_config"] += 1
            continue
        # Checked before deduplication too: one run must use one panel.
        digest_here = result.get("persona_digest")
        if digest_here:
            if persona_digest and digest_here != persona_digest:
                raise ValueError(f"{trial} used a different persona panel from earlier trials of "
                                 f"this config; collect each panel's jobs separately")
            persona_digest = digest_here
        focal = _focal_name(trial_config.get("agent") or {})
        key = (cell["seed"], focal, cell["arm"], cell["replicate"])
        if key in rows:
            counts["duplicates"] += 1
            continue
        focals[focal] = {"name": focal, "provider": "harbor", "model": focal}
        cell_key = f"{run_id}-{cell['seed']}-{focal}-{cell['arm']}-r{cell['replicate']}"
        trace_rel = f"traces/{cell_key}.jsonl"
        done = bool(result.get("completed") and result.get("verified"))
        lines = read_trace(trial / "verifier" / "trace.jsonl") if done else []
        if done:
            copies.append((trial / "verifier" / "trace.jsonl", trace_rel))
        rows[key] = {"type": "cell", "cell_key": cell_key,
                     "pair_key": f"{run_id}-{cell['seed']}-{focal}", "arm": cell["arm"],
                     "seed": cell["seed"], "focal": focal, "focal_seat": None,
                     "replicate": cell["replicate"], "seats": cell.get("seats") or {},
                     "status": "done" if done else "errored", "trace": trace_rel if done else None,
                     "usage": _usage(lines), "harbor_trial": str(trial)}
        if not done:
            rows[key]["error"] = "; ".join(result.get("problems") or []) or "not completed"
        counts["kept"] += 1
    config = json.loads(json.dumps(config))
    config["lineup"] = [*config["lineup"], *focals.values()]
    config["focal_models"] = sorted(focals)
    run = {"type": "run", "run_id": f"{run_id}-harbor", "config_hash": chash, "config": config,
           "source": "harbor"}
    if persona_digest:
        run["persona_digest"] = persona_digest
    _write_run(out, [run, *rows.values()], copies)
    return counts


def _write_run(out: Path, rows: list[dict[str, Any]], copies: list[tuple[Path, str]]) -> None:
    """Write a run directory so a failure leaves the previous one intact: the
    traces go to a staging directory that replaces `traces/` only when all are
    copied, then the manifest is replaced atomically. Nothing else in `out` is
    touched."""
    out.mkdir(parents=True, exist_ok=True)
    staging, old = out / "traces.collecting", out / "traces.previous"
    for d in (staging, old):
        if d.exists():
            shutil.rmtree(d)
    staging.mkdir()
    for src, rel in copies:
        shutil.copy(src, staging / Path(rel).name)
    tmp = out / "manifest.jsonl.collecting"
    with tmp.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")
    if (out / "traces").exists():
        (out / "traces").rename(old)
    staging.rename(out / "traces")
    os.replace(tmp, out / "manifest.jsonl")
    if old.exists():
        shutil.rmtree(old)
