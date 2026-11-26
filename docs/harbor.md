# Harbor export (experimental)

`parley-sim harbor-export` turns a matched-run config into
[Harbor](https://github.com/harbor-framework/harbor) tasks, so any Harbor
agent (Claude Code, Codex, Terminus, your own) can play the focal seat against
the same counterparts and seeds as a local run. `parley-sim harbor-collect`
brings the results back for `parley-sim report`.

Experimental in 0.0.x: tested end to end with Harbor 0.27 on local Docker,
using Harbor's oracle agent and scripted counterparts. It hasn't yet been run
with a model as the agent or as the counterpart.

## How a task works

One task plays one episode: one cell of the config's grid (seed, arm,
replicate). The config's `focal_models` are not used; the agent under test is
whatever you run the tasks with.

- **Sidecar.** A `parley-runtime` service holds the episode and every
  counterpart seat: a scripted policy, a model, or a UserSim persona model. The
  counterpart's private values, model and API key exist only there.
- **Tools.** The agent gets two MCP tools: `start_negotiation` (role, private
  information, rules, the counterpart's opening) and `take_turn` (a message,
  one move, a short explanation). A move that doesn't parse is a strike, as for
  a model seat.
- **Scoring.** The verifier fetches the result from the sidecar with a
  per-task token that only the sidecar and the verifier hold. The sidecar
  replays the trace (`verify`) first. `reward.json`:

| key | meaning |
|---|---|
| `reward` | `focal_surplus_share`, as in local runs (negative for a deal below the agent's walk-away). 0 if the episode didn't finish or verify. |
| `no_deal` | 1 when the episode finished without a deal (its reward is 0) |
| `completed` | 1 when the episode reached its end |
| `verified` | 1 when the trace replays exactly |

The verifier also saves the full trace (`verifier/trace.jsonl`) and the
episode's metrics (`verifier/result.json`).

## Export

```bash
uv build --wheel
parley-sim harbor-export configs/equiv-sd3.yaml --out tasks --wheel dist/parley_sim-0.0.1-py3-none-any.whl
```

Each task builds the sidecar from the wheel. To build it once instead, use
the image in `integrations/harbor/Dockerfile` and pass `--image`:

```bash
docker build -f integrations/harbor/Dockerfile -t parley-runtime .
parley-sim harbor-export CONFIG.yaml --out tasks --image parley-runtime
```

Add `--build-arg EXTRAS=harbor,openai,anthropic,usersim` for a persona arm.

## Run

```bash
export OPENROUTER_API_KEY=...   # for model counterparts: the key variables their lineup entries use
harbor run -p tasks -a claude-code -m anthropic/claude-sonnet-5 -n 4 -o jobs
```

- **Keys.** Model counterparts read their key from the variable their lineup
  entry names (`api_key_env`, or the provider's default); the task passes that
  variable from your shell to the sidecar only.
- **Persona arm.** Tasks with a UserSim persona counterpart mount your own
  panel: set `PARLEY_PERSONA_PANEL` to the parquet from `usersim panel` (see
  [personas.md](personas.md)). No persona data is written into the tasks.
- **Timeouts.** A `take_turn` call returns after every counterpart has
  answered, which can take up to twice each counterpart's `timeout_s`. Agents
  whose MCP client times out sooner need a longer tool timeout.

## Report

```bash
parley-sim harbor-collect jobs --config CONFIG.yaml --out run
parley-sim report run
```

Each Harbor agent and model becomes a focal model (for example
`claude-code.anthropic-claude-sonnet-5`), compared across arms on the same
seeds as a local run. Only the first trial of each task per agent is kept.
Counterpart token use is counted; the agent's own usage is in Harbor's results.

## Limits

- **Public scenarios.** A seed's scenario, including the counterpart's
  walk-away, is computable from the public generator. The agent's container
  never sees the seed, but a published task names it. For a private
  evaluation, export with `--seed-secret-env VAR` (the same secret for every
  task, so arms stay matched): scenarios then can't be computed from the seed.
- **Prompts.** The agent sees the same role, rules and observations as a
  model seat, but answers through `take_turn` instead of a JSON reply, so
  Harbor results compare agents with each other, not with local runs.
- **Docker only so far.** Tasks use a sidecar (Docker Compose), which Harbor
  supports on local Docker and some cloud sandboxes.
