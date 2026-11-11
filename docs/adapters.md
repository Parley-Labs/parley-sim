# Adapters

The episode runtime never calls a model itself. Whenever a model seat must
act, it hands back a decision request and waits for the answer:

```python
event = episode.advance()
while not isinstance(event, EpisodeComplete):
    event = episode.advance(answer(event))
```

The local runner, the NeMo Gym server and the OpenEnv environment are each a
thin loop around that call, so an episode behaves the same, and writes the
same trace, wherever it runs. Both adapters work with any environment
parley-sim can load, including [your own](custom-environments.md).

Status: both adapters are covered by unit tests against the pinned library
versions, but have not yet been exercised against a production Gym or
OpenEnv deployment. Reports of what breaks are welcome.

## NeMo Gym

`pip install "parley-sim[gym]"`, targeting nemo-gym 0.7 (Python 3.13.14 or
newer, as nemo-gym requires).

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

## OpenEnv

`pip install "parley-sim[openenv]"`, targeting openenv 0.12.

Your client (a trainer such as TRL's GRPOTrainer, or a script) plays the
focal seat through `reset` and `step`. Counterparts, scripted or model, run
inside the environment, and the full trace, including every counterpart
decision, is exposed through `state()`, so opponent behavior is never hidden
in the container.

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

Provider keys for model counterparts are read from the container's
environment at runtime (`docker run -e OPENROUTER_API_KEY ...`); never bake
them into the image.

A reply that doesn't parse, or an illegal move, is recorded as a strike and
replaced by the environment's fallback move, exactly as in a local run.

To compare arms, serve one container per kind of counterpart and run the same
seeds through each. Training against one kind of counterpart only hides how
the agent behaves against the other.
