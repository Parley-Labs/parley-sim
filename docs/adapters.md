# Adapters

The episode runtime never calls a model itself. Whenever a model seat must
act, it hands back a decision request and waits for the answer:

```python
event = episode.advance()
while not isinstance(event, EpisodeComplete):
    event = episode.advance(answer(event))
```

The local runner, the NeMo Gym server and the OpenEnv environment are each a
thin loop around that call, so the rules, scenarios, scripted counterparts and
trace format are the same wherever an episode runs. Both adapters work with
any environment parley-sim can load, including
[your own](custom-environments.md). How seat failures are handled differs:

| | Local runner | NeMo Gym | OpenEnv |
| --- | --- | --- | --- |
| Malformed reply from a model seat | One retry, then a strike | One retry, then a strike | Focal seat: a strike at once (your client owns retries); counterparts: as local |
| Timeout or provider error | Strike, fallback move; a 400 or 422, or three failed turns in a row from one seat, fails the episode; a 401, 402, 403 or 404 halts the run | The episode fails (retryable per Gym's rules) | Counterparts: as local |
| Refusal breaker scope | The whole run | One episode | One session (all its episodes) |

Status: both adapters are covered by unit tests against the pinned library
versions. The OpenEnv package has also been built as a Docker image and
played end to end through its WebSocket client. The NeMo Gym adapter has not yet run against a
live Gym deployment with real agent servers; its training payload is tested
through Gym's own rollout-record code, not yet through a NeMo-RL training run. Reports of what breaks are
welcome.

## NeMo Gym

Install the `gym` extra from a checkout (`uv sync --extra gym --extra openai`),
targeting nemo-gym 0.7 (Python 3.13.14 or newer, as nemo-gym requires).

One Gym `/run` is one parley-sim episode. Scripted seats run in-process.
Every model seat's turn goes to a Gym agent server through an agent session;
only the focal seat's calls use Gym's token-capture route, so training data
contains the agent under training and nothing else. The episode's reward is
an environment metric (default `focal_surplus_share`), and the full trace,
per-seat outcomes and every metric ride along in the result.

To serve it from a Gym checkout, add `environment_servers/parley_sim/app.py`:

```python
from nemo_gym.server_utils import is_nemo_gym_fastapi_entrypoint
from parley_sim.adapters.nemo_gym import ParleySimEnvironmentServer

if __name__ == "__main__":
    ParleySimEnvironmentServer.run_webserver()
elif is_nemo_gym_fastapi_entrypoint(__file__):
    app = ParleySimEnvironmentServer.run_webserver()
```

and `environment_servers/parley_sim/configs/parley_sim.yaml`:

```yaml
parley_sim:
  environment_servers:
    parley_sim:
      entrypoint: app.py
      agent_server:        # the focal agent
        type: responses_api_agents
        name: ???
      peer:                # a counterpart, serving "model:peer" seats
        type: responses_api_agents
        name: ???
      reward_metric: focal_surplus_share
      max_refusals: 3
      cleanup_timeout_seconds: 180
      default_episode_timeout_seconds: 3600

environment_server_routes:   # taskset -> environment server
  parley_sim: parley_sim
```

| Server key | Meaning |
| --- | --- |
| `agent_server` | The focal agent: the one under training or evaluation. Gym attributes this server's results to `agent_server`, so token-capture finalisation follows the focal agent's `token_id_capture` setting. |
| `<name>` | Any other key holding a `responses_api_agents` reference is a counterpart, used by `model:<name>` seats. Any other unknown key is an error. |
| `reward_metric` | The environment metric used as the reward (default `focal_surplus_share`) |
| `max_refusals` | Refusals in one episode before it halts (default 3) |
| `focal_conversation` | Play the focal seat as one growing conversation (default `true`; see [Training data](#training-data)) |
| `allowed_env_paths` | Import paths task rows may name as `env` (default none) |
| `seed_secret_env` | Name of an environment variable holding a secret that keys task seeds into scenario seeds (default none); see [seeds and secrecy](concepts.md#seeds-and-secrecy) |
| `actor_call_timeout_seconds` | Per-call timeout for agent requests (default 300) |
| `responses_create_params` | Merged into every Responses API request, e.g. `{reasoning: {effort: low}}` |

Counterparts are top-level keys, not a nested map, because Gym treats only
top-level `type: responses_api_agents` fields as agent references. That way
Gym checks each one names a defined agent and does not generate a legacy relay
server for it, so the config loads with
`error_on_agent_without_environment_server: true`. An agent may serve as both
`agent_server` and a counterpart (self-play); rows routed by taskset, as below,
are unaffected.

Gym's collector sends an episode to an environment server only for a
materialized row: `task_id` names a taskset, `task_input` is the parley-sim
task, and `environment_server_routes` maps the taskset to this server. Each
row is one episode (one line of the JSONL; wrapped here):

```json
{"task_id": {"taskset": "parley_sim", "task_id": "s1000-hardliner"},
 "task_input": {"seed": 1000, "seats": {"agent": "model:focal", "customer": "policy:hardliner"},
                "arm": "soc", "pair_key": "run-1000-focal"}}
```

A flat row (the `task_input` alone) is not an episode request and gets a
422: Gym builds the request's episode id, which carries the token-capture
key, only for a materialized row.

| `task_input` field | Meaning |
| --- | --- |
| `seed` | Scenario seed |
| `seats` | Seat map: `model:<any name>` on the focal seat for the agent under training, `policy:<name>` or `model:<counterpart>` for the rest |
| `env`, `env_config`, `focal_seat` | As in a run config |
| `arm`, `pair_key`, `cell_key`, `replicate` | Labels recorded in the trace, so Gym rollouts can be paired and reported like a parley-sim run |

To compare arms, write task rows for the same seeds with each kind of
counterpart, and give the matching rows the same `pair_key`.

Task rows can name only built-in, registered or entry-point environments. An
import path (`my_pkg.envs:ENTRY`) in a task row is refused unless the server
config lists it under `allowed_env_paths`, because loading one imports and
runs code named by the task data. These fail at seeding, as terminal
failures, before any agent is called: an unknown environment, reward metric,
counterpart or `focal_seat`, and an allowed import path that does not load
(missing module or attribute, or not an `EnvEntry`).

The reward metric must be defined for the episode. Some metrics are undefined
for some episodes (a share of an offer that was never made); such an episode
fails at the `verification` stage as a terminal failure with
`failure_kind: parley_sim:reward_undefined`, rather than being scored 0.

If the adapter cannot process a reply the agent did return, the episode fails
as a terminal `agent` failure instead of being scored on fallback moves.

### Training data

The focal seat plays one growing conversation (`focal_conversation`, on by
default): each request resends the earlier turns, with the agent's own output
items exactly as it returned them (unset fields stay unset, as Gym's
SimpleAgent replays them), then adds what the counterpart did since. Each
prompt therefore extends the previous one at the message level. Token
continuity, which NeMo Gym needs to treat an episode as one continuous
training sample, also requires the agent's model server to tokenize the
resent output items into the same token ids it sampled. The adapter does not
check that; each output item's `prompt_token_ids` let a trainer verify it.

A refused call (any refusal part, even beside text), or an empty or
whitespace-only reply, never enters the conversation. A refusal is a strike. An empty reply is retried once with a
correction appended to the same prompt, as in a local run. Both are still
listed in `focal_calls`.

The episode result carries:

- `response`: a Gym-native Responses object whose `output` is every focal
  reply in the conversation, in order, with whatever token ids and log
  probabilities the agent returned. Counterpart replies and refused or empty
  calls are never in it. Its envelope, `id` included, is the last such
  reply's, so Gym's token-capture finaliser can match it to the captured
  call. Gym's rollout collector stores it as the rollout's `response`, so
  `rollout_carries_token_ids` sees the agent's sampled tokens and token
  capture leaves them as they are.
- `mask_sample`: Gym's flag for a sample the trainer must skip. It is set
  when some focal output items carry generated token ids and others don't:
  Gym keeps inline ids as they are and fills in none, so the sample would
  otherwise train on only part of the agent's tokens.
- `focal_calls`: every focal request and full response, for audit, each
  flagged `refused`, `empty` and `in_conversation`. Gym's own `_ng_*` keys,
  such as `_ng_trajectory` with observability on, are dropped from both.

A reply cut off by the agent's output cap (`status: incomplete`,
`incomplete_details.reason: max_output_tokens`) that does not parse is
recorded as a `truncated` strike, without a repair retry.

With `focal_conversation: false` the focal seat gets a fresh prompt each turn,
like a default local run, the result carries no training payload, and the
focal calls do not use the token-capture route. This mode is for evaluation:
set `token_id_capture: false` on the focal agent for it. Gym's collector
finalises token capture for every rollout attributed to an agent with capture
on; when it rebuilds responses from captured tokens, it would mask each such
rollout as having no recorded tokens, which counts against
`token_id_capture.max_mask_fraction`. To evaluate locally
with the prompt shape used in training, set `conversation: true` on the focal
model in the run config.

Tested against the pinned nemo-gym 0.7 code path from episode to collected
rollout record; not yet with a NeMo-RL training run.

## OpenEnv

Install the `openenv` extra from a checkout (`uv sync --extra openenv
--extra openai`), targeting openenv 0.12.

Your client (a trainer such as TRL's GRPOTrainer, or a script) plays the
focal seat through `reset` and `step`. Counterparts, scripted or model, run
inside the environment. While an episode runs, `state()` shows only what the
focal seat can see. Once an episode ends, its full trace, including the
scenario, ground truth and every counterpart decision, is released, so
opponent behavior is never hidden in the container. The client chooses the
seeds, so it can end an episode at once, read the released trace, and replay
the same seed knowing the counterpart's private values. With
`PARLEY_RELEASE_TRACE=never`, `state()` only ever shows what the focal seat
saw, plus the outcomes and metrics, so counterpart decisions (and an LLM
counterpart's messages beyond what the focal seat saw) stay in the server.
It does not make the scenario secret: the scenario generator is public and
deterministic, so a client that picks a seed can compute the counterpart's
private values itself. Treat the client that drives training as trusted; the
model being trained only ever sees the observations the client passes it.

Play goes only through OpenEnv's WebSocket sessions (`/ws`, which
`ParleySimEnv` uses), where the session limit, idle timeout and refusal
breaker apply. The server serves only `/ws`, `/health`, `/schema`,
`/metadata` and the API docs; every other OpenEnv route is dropped, since most
of them build an environment per request outside those limits. The stateless
`/reset`, `/step`, `/state` and `/mcp` routes answer 405 with a pointer to
`/ws`, and OpenEnv's web interface is never mounted, whatever
`ENABLE_WEB_INTERFACE` says, because it would share one environment between
all its users. If you deploy with `openenv push`, pass `--no-interface`.

`ParleySimEnv` waits up to 300 seconds for each reply (`message_timeout_s`;
OpenEnv's default is 60), since one model counterpart turn can take two
attempts of its `timeout_s` (120 seconds by default). Raise it if you raise
`timeout_s`. A focal reply may be up to 65,536 characters; a structured `move`
keeps only the keys the action schema declares.

The configuration (seat labels, lineup names, policies, reward metric,
provider keys) is checked when the server starts, so a typo fails startup
instead of every reset.

The server has no authentication, and with model counterparts every session
spends your provider keys, so expose it only on a trusted network.
`uv run --project . server`, run from `integrations/openenv/parley_sim_env`
(`--project .` is the current directory), binds to localhost by default. Idle
sessions close after `PARLEY_SESSION_TIMEOUT` seconds (default 600), so
abandoned sessions can't hold every slot. The server refuses to start if that
is shorter than twice the largest `timeout_s` of a seated lineup model, since
the idle check could otherwise close a session in the middle of a step; leave
room for your own model's time per turn as well. OpenEnv 0.12 counts idle
time from the end of the last `reset` or `step`, not its start, so a step sent
just before the deadline can still be closed while a counterpart is
answering: keep the timeout well above the longest pause between your
client's steps plus one counterpart decision. Only `reset` and `step` count
as activity; a client that only polls `state()` is closed as idle. OpenEnv
0.12 also frees a reaped session's slot while its connection is still open;
parley-sim closes the environment, so that connection only gets errors.
Both are OpenEnv bugs, written up with a repro in
[known issues](known-issues.md#openenv-012-idle-reaper-closes-sessions-mid-step).

A ready-to-build package is in
[`integrations/openenv/parley_sim_env`](../integrations/openenv/parley_sim_env/README.md),
with its own README, `openenv.yaml` and Dockerfile:

```bash
docker build -f integrations/openenv/parley_sim_env/server/Dockerfile -t parley-sim-env .
docker run -p 127.0.0.1:8000:8000 parley-sim-env
```

The image listens on all interfaces inside the container;
`-p 127.0.0.1:8000:8000` publishes the port on this machine only. There is no
authentication, so publish it more widely only on a trusted network.

Play an episode:

```python
from parley_sim.adapters.openenv import ParleySimAction, ParleySimEnv

with ParleySimEnv(base_url="http://localhost:8000").sync() as env:
    result = env.reset(seed=1000)
    while not result.done:
        obs = result.observation
        reply = my_model(obs.system_prompt, obs.user_prompt)  # JSON per obs.action_schema
        result = env.step(ParleySimAction(text=reply))
    print(result.reward, env.state().metrics)
```

The server reads its configuration from environment variables:

| Variable | Default | Meaning |
| --- | --- | --- |
| `PARLEY_ENV` | `support_desk` | Environment name or import path |
| `PARLEY_SEATS` | `{"agent": "client", "customer": "policy:reasonable"}` | Seat map; the focal seat is `client` |
| `PARLEY_LINEUP` | `[]` | Model specs for `model:<name>` seats ([format](configuration.md#model-specs)) |
| `PARLEY_ENV_CONFIG` | `{}` | Environment options |
| `PARLEY_REWARD_METRIC` | `focal_surplus_share` | Which episode metric is the reward |
| `PARLEY_MAX_REFUSALS` | `3` | Stop a session after this many provider refusals |
| `MAX_CONCURRENT_ENVS` | `8` | Concurrent sessions |
| `PARLEY_SESSION_TIMEOUT` | `600` | Close sessions idle this many seconds; at least 2 × the largest seated `timeout_s` |
| `PARLEY_RELEASE_TRACE` | `after_done` | `after_done`: full trace in `state()` once an episode is done; `never`: focal view plus outcomes and metrics only |
| `PARLEY_SEED_SECRET` | none | Keys requested seeds into scenario seeds, so a client can't compute a scenario from its seed; use with `PARLEY_RELEASE_TRACE=never` ([seeds and secrecy](concepts.md#seeds-and-secrecy)) |

Provider keys for model counterparts are read from the container's
environment at runtime
(`docker run -p 127.0.0.1:8000:8000 -e OPENROUTER_API_KEY ...`); never bake
them into the image.

A reply that doesn't parse, or an illegal move, is recorded as a strike and
replaced by the environment's fallback move, exactly as in a local run.

To compare arms, serve one container per kind of counterpart and run the same
seeds through each. Training against one kind of counterpart only hides how
the agent behaves against the other.
