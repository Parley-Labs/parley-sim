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
| Timeout or provider error | Strike, fallback move | The episode fails (retryable per Gym's rules) | Counterparts: strike, fallback move |
| Refusal breaker scope | The whole run | One episode | One session (all its episodes) |

Status: both adapters are covered by unit tests against the pinned library
versions. The OpenEnv package has also been built as a Docker image and
played end to end over HTTP. The NeMo Gym adapter has not yet run against a
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
      focal_agent: {type: responses_api_agents, name: ???}
      counterpart_agents:              # used as "model:<name>" seats
        peer: {type: responses_api_agents, name: ???}
      reward_metric: focal_surplus_share
      max_refusals: 3
      cleanup_timeout_seconds: 180
      default_episode_timeout_seconds: 3600
```

Each task row is one episode:

```json
{"seed": 1000, "seats": {"agent": "model:focal", "customer": "policy:hardliner"},
 "arm": "soc", "pair_key": "run-1000-focal"}
```

| Task field | Meaning |
| --- | --- |
| `seed` | Scenario seed |
| `seats` | Seat map: `model:focal` for the agent under training, `policy:<name>` or `model:<counterpart>` for the rest |
| `env`, `env_config`, `focal_seat` | As in a run config |
| `arm`, `pair_key`, `cell_key`, `replicate` | Labels recorded in the trace, so Gym rollouts can be paired and reported like a parley-sim run |

To compare arms, write task rows for the same seeds with each kind of
counterpart, and give the matching rows the same `pair_key`.

### Training data

The focal seat plays one growing conversation (`focal_conversation`, on by
default): each request resends the earlier turns, with the agent's own output
items exactly as it returned them, then adds what the counterpart did since.
Each prompt therefore extends the previous one at the message level. Token
continuity, which NeMo Gym needs to treat an episode as one continuous
training sample, also requires the agent's model server to tokenize the
resent output items into the same token ids it sampled. The adapter does not
check that; each output item's `prompt_token_ids` let a trainer verify it.

The episode result carries:

- `response`: a Gym-native Responses object whose `output` is every focal
  reply in order, with whatever token ids and log probabilities the agent
  returned. Counterpart replies are never in it. Gym's rollout collector
  stores it as the rollout's `response`, so `rollout_carries_token_ids` sees
  the agent's sampled tokens and token capture leaves them as they are.
- `focal_calls`: every focal request and full response, for audit.

With `focal_conversation: false` the focal seat gets a fresh prompt each turn,
like a default local run, and the result carries no training payload. To
evaluate locally with the prompt shape used in training, set
`conversation: true` on the focal model in the run config.

Tested against the pinned nemo-gym 0.7 code path from episode to collected
rollout record; not yet with a NeMo-RL training run.

Task rows can name only built-in, registered or entry-point environments. An
import path (`my_pkg.envs:ENTRY`) in a task row is refused unless the server
config lists it under `allowed_env_paths`, because loading one imports and
runs code named by the task data. The reward metric is checked against the
environment's metrics, and a task with an unknown one fails at seeding rather
than scoring 0.

## OpenEnv

Install the `openenv` extra from a checkout (`uv sync --extra openenv
--extra openai`), targeting openenv 0.12.

Your client (a trainer such as TRL's GRPOTrainer, or a script) plays the
focal seat through `reset` and `step`. Counterparts, scripted or model, run
inside the environment. While an episode runs, `state()` shows only what the
focal seat can see; once it ends, the full trace, including the scenario,
ground truth and every counterpart decision, is released, so opponent
behavior is never hidden in the container and never leaked mid-episode.

The configuration (seat labels, lineup names, policies, reward metric,
provider keys) is checked when the server starts, so a typo fails startup
instead of every reset.

The server has no authentication, and with model counterparts every session
spends your provider keys. `uv run --project . server` binds to localhost by
default. Idle sessions close after `PARLEY_SESSION_TIMEOUT` seconds (default
600), so abandoned sessions can't hold every slot.

A ready-to-build package is in
[`integrations/openenv/parley_sim_env`](../integrations/openenv/parley_sim_env/README.md),
with its own README, `openenv.yaml` and Dockerfile:

```bash
docker build -f integrations/openenv/parley_sim_env/server/Dockerfile -t parley-sim-env .
docker run -p 8000:8000 parley-sim-env
```

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
| `PARLEY_SESSION_TIMEOUT` | `600` | Close sessions idle this many seconds |

Provider keys for model counterparts are read from the container's
environment at runtime (`docker run -e OPENROUTER_API_KEY ...`); never bake
them into the image.

A reply that doesn't parse, or an illegal move, is recorded as a strike and
replaced by the environment's fallback move, exactly as in a local run.

To compare arms, serve one container per kind of counterpart and run the same
seeds through each. Training against one kind of counterpart only hides how
the agent behaves against the other.
