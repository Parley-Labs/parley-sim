# Harbor export (experimental)

`parley-sim harbor-export` turns a matched-run config into
[Harbor](https://github.com/harbor-framework/harbor) tasks, so any Harbor
agent (Claude Code, Codex, Terminus, your own) can play the focal seat against
the same counterparts and seeds as a local run. `parley-sim harbor-collect`
brings the results back for `parley-sim report`.

Experimental in 0.0.x. Tested end to end with Harbor 0.27 on local Docker,
and used for a pre-registered run: 400 episodes with Claude Code and
Terminus-2 as agents against scripted, plain Claude and UserSim persona
customers, every trace verified. Results:
[`results/README.md`](../results/README.md#harbor-pilot-pre-registered).

## How a task works

One task plays one episode: one cell of the config's grid (seed, arm,
replicate). The config's `focal_models` are not used; the agent under test is
whatever you run the tasks with.

- **Any environment.** The export works for any parley-sim environment
  (built in, or [your own](custom-environments.md) installed in the sidecar
  image), not only `support_desk`.
- **Sidecar.** A `parley-runtime` service holds the episode and every
  counterpart seat: a scripted policy, a model, or a UserSim persona model. The
  counterparts' private information, models and API keys exist only there.
  Scripted counterparts are optional: use them for cheap mock runs and as a
  baseline arm ([concepts](concepts.md#arms)).
- **Tools.** The agent gets two MCP tools: `start_episode` (role, rules,
  private information, what has happened so far) and `take_turn` (one action;
  its input schema is the environment's action schema). An action that doesn't
  parse is a strike, as for a model seat.
- **Scoring.** The verifier fetches the result from the sidecar with a
  per-task token that only the sidecar and the verifier hold. The sidecar
  replays the trace (`verify`) first. `reward.json`:

| key | meaning |
|---|---|
| `reward` | The episode metric chosen with `--reward-metric` (default `focal_surplus_share`, as in local runs; negative for a deal below the agent's walk-away). 0 if the episode didn't finish or verify, or the metric is undefined. |
| `no_deal` | 1 when the episode finished without a deal (with the default metric, its reward is then 0). Only for environments with a `deal` metric. |
| `completed` | 1 when the episode reached its end |
| `verified` | 1 when the trace replays exactly |

The verifier also saves the full trace (`verifier/trace.jsonl`) and the
episode's metrics (`verifier/result.json`). Each task's `solution/solve.sh`
(Harbor's oracle agent) is a plumbing check, not a solution: it submits the
smallest action the schema allows each turn and plays the episode to its end.

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
- **Image builds.** Docker Hub rate limits can fail a task's build before the
  episode starts (no model calls are made). Re-run those tasks;
  `harbor-collect` keeps only trials that reached the verifier.
- **Timeouts.** A `take_turn` call returns after every counterpart has
  answered, which can take up to twice each counterpart's `timeout_s`. Agents
  whose MCP client times out sooner need a longer tool timeout.

## Harbor's simulated user as the counterpart (experimental)

Instead of a counterpart inside the sidecar, Harbor's own simulated user can
play the counterpart seat. Any Harbor agent can then be the counterparty, with
Harbor's `--user-persona-path` setting its personality, while the environment
still enforces the rules and scores the outcome exactly.

```bash
parley-sim harbor-export configs/meeting-canary.yaml --out tasks --image parley-runtime \
  --harbor-user-arm harbor-user
```

This adds an arm, `harbor-user`, on the same seeds. Its tasks differ in three ways:

- Their `instruction.md` is the counterparty's brief. In a simulated-user
  trial Harbor gives the instruction to the user agent only, never to the
  agent under test.
- The sidecar also serves `counterpart_start` and `counterpart_take_turn`.
  They need a per-task counterparty token, which only that brief contains.
- Both sides act only through their tools, so every move and message is in the
  trace. The bridge is used only to tell the agent under test it's their turn.

Run those tasks with a simulated user that can reach the sidecar, using a job
config:

```yaml
user_agent:
  name: claude-code
  model_name: anthropic/claude-haiku-5-5
  bridge: {kind: acp}
  mcp_servers:
    - {name: parley-runtime, transport: streamable-http, url: http://parley-runtime:8000/mcp}
```

Every tool response to the agent under test says that the chat prompting it is
the other side, and to reply there only with "Done, your turn." Both sides' turn
tools say plainly when an action was not recorded (not their turn) or could not
be used (a strike).

Caveats:
- **Small runs only so far.** 19 harbor-user trials on `meeting_scheduling`
  (Claude Code against Claude Code on Haiku 5.5), all completed and verified;
  turn-taking over the bridge worked in every one. Not yet a study.
- **Supported agents.** Harbor's bridge supports only some agents under test
  (Claude Code, Codex, Gemini CLI, OpenCode, Copilot CLI).
- **Chat leaks.** Harbor sends everything the agent under test writes to the
  simulated user, and the sidecar never sees it, so chat is not in the trace.
  `harbor-collect` reads it from Harbor's bridge log
  (`agent/bridge-trajectory.json`): each simulated-user trial's manifest row
  gets `chat` (the agent's replies, and how many leak private information when
  the environment can tell), and the totals are printed. For
  `meeting_scheduling` a leak is a private slot rating stated as a number.
  The log format is the ACP client's session export; the parser follows what
  Harbor 0.27 wrote, so check the counts on a new Harbor version.
- **One simulated user per arm.** Harbor's `user_agent` settings pick the
  counterparty, outside the config hash. `harbor-collect` records them (agent,
  model, persona, bridge) as the arm's definition. The persona is identified by
  the digest Harbor recorded in each trial's `lock.json` when it ran (or, for an
  older Harbor, by reading the file, so run collect where the persona files
  are); a persona it can't identify is refused. It also refuses to
  mix trials of one arm played by different simulated users, and `report` refuses to
  join runs whose arm of the same name was played differently. To compare
  simulated users, export each as its own arm (`--harbor-user-arm NAME`).
- **Usage figures.** In these trials, Harbor's per-trial usage (`agent_result`)
  is the simulated user's, not the agent under test's; read the agent's usage
  from its own logs.
- **Token.** The two agents share one container, so the counterparty token is
  a guard against mistakes, not a security boundary.

## Persona sweeps over Harbor's own simulated-user tasks (experimental)

Harbor can run any task with a simulated user (`--user-agent`,
`--user-persona-path`): the user agent gets the instruction and talks to the
agent under test, and the task's verifier scores the result. `harbor-sweep`
runs the same tasks under different personas and asks whether the agent's
reward changes. The task is the matched unit, as the seed is in a local run.

```bash
parley-sim harbor-sweep personas --panel PANEL.parquet --n 5 --out personas   # keep private
parley-sim harbor-sweep plan --tasks TASK... --personas personas --include-default \
  --agent claude-code --model anthropic/claude-sonnet-5-5 \
  --user-agent claude-code --user-model anthropic/claude-haiku-5-5 --jobs jobs > sweep.sh
bash sweep.sh   # runs the trials
parley-sim harbor-sweep report jobs --baseline harbor-default --margin 0.1
```

Arms come from persona file names: `harbor-default` (no file), `usersim`
(`usersim-sNN.md`, from a UserSim panel), `minimal` (a terse user), or any
`<arm>.md` you write. The files `personas` writes, `minimal.md` included,
start with the same rules for staying in character. Persona files built from a panel hold persona data; keep
them out of version control. Not yet run with real agents.

## Report

```bash
parley-sim harbor-collect jobs --config CONFIG.yaml --out run
parley-sim report run
```

Each Harbor agent and model becomes a focal model (for example
`claude-code.anthropic-claude-sonnet-5`), compared across arms on the same
seeds as a local run. Only trials of tasks exported from this exact config
(same config hash, recorded in each result) are used; trials of an edited
config, even with the same `run_id`, are skipped and counted. Then only the
first trial of each task per agent is kept.
Counterpart token use is counted; the agent's own usage is in Harbor's results.

## Limits

- **Public scenarios.** A seed's scenario, including the counterpart's
  walk-away, is computable from the public generator. The agent's container
  never sees the seed, but a published task names it. For a private
  evaluation, export with `--seed-secret-env VAR` (the same secret for every
  task, so arms stay matched): scenarios then can't be computed from the seed.
  The secret is not written into the tasks; set `VAR` in the shell that runs
  them, as for model keys.
- **Prompts.** The agent sees the same role, rules and observations as a
  model seat, but answers through `take_turn` instead of a JSON reply, so
  Harbor results compare agents with each other, not with local runs.
- **Docker only so far.** Tasks use a sidecar (Docker Compose), which Harbor
  supports on local Docker and some cloud sandboxes.
