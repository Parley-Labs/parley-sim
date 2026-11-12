# parley-sim for OpenEnv

Negotiation environments from [parley-sim](../../../README.md). Your model plays
the **agent**; the environment plays the **counterpart**: a scripted persona or
another LLM. Every episode has exact ground truth (all possible deals are
enumerable), so rewards are computed, not judged.

The demo environment, `support_desk`, is a retention negotiation. A customer
wants to cancel; your model is the support agent with private authorization
limits, negotiating discount, credit, upgrade and billing term.

## Run it

Locally, from this folder:

```bash
uv run --project . server            # http://localhost:8000
```

With Docker, from the parley-sim repository root:

```bash
docker build -f integrations/openenv/parley_sim_env/server/Dockerfile -t parley-sim-env .
docker run -p 8000:8000 parley-sim-env
```

## Play an episode

```python
from parley_sim_env import ParleySimAction, ParleySimEnv

with ParleySimEnv(base_url="http://localhost:8000").sync() as env:
    result = env.reset(seed=1000)
    while not result.done:
        obs = result.observation
        reply = my_model(obs.system_prompt, obs.user_prompt)  # JSON move as text
        result = env.step(ParleySimAction(text=reply))
    print(result.reward)            # agent's share of the surplus it could have won
    print(env.state().metrics)      # every judge-free metric for the episode
```

A reply that doesn't parse, or proposes a deal the agent isn't authorized to
offer, is recorded as a strike and replaced by a no-op move, exactly as in
parley-sim's own runs. `env.state()` also carries the full episode trace,
including the counterpart's moves.

## Configure the counterpart

The server reads its configuration from environment variables:

| Variable | Default | Meaning |
|---|---|---|
| `PARLEY_ENV` | `support_desk` | Environment name or import path |
| `PARLEY_SEATS` | `{"agent": "client", "customer": "policy:reasonable"}` | Who plays each seat. `client` is your model. |
| `PARLEY_LINEUP` | `[]` | Model specs for `model:<name>` seats. |
| `PARLEY_ENV_CONFIG` | `{}` | Environment options, e.g. `{"max_turns": 10}`. |
| `PARLEY_REWARD_METRIC` | `focal_surplus_share` | Which episode metric is the reward. |
| `PARLEY_MAX_REFUSALS` | `3` | Stop a session after this many provider refusals. |
| `MAX_CONCURRENT_ENVS` | `8` | Concurrent sessions. |
| `PARLEY_SESSION_TIMEOUT` | `600` | Close sessions idle this many seconds. |

The server has no authentication, and with model counterparts every session
spends your provider keys. `uv run --project . server` binds to localhost; the
Docker image listens on all interfaces inside the container, so publish its
port only on a trusted network.

During an episode, `env.state()` shows only what your model's seat can see.
The full trace, with the scenario, ground truth and the counterpart's
decisions, is available once the episode is done.

Scripted customers: `policy:hardliner`, `policy:reasonable`, `policy:conceder`.
An LLM customer needs a lineup entry and its provider key at runtime:

```bash
docker run -p 8000:8000 -e OPENROUTER_API_KEY \
  -e PARLEY_SEATS='{"agent":"client","customer":"model:peer"}' \
  -e PARLEY_LINEUP='[{"name":"peer","provider":"openai_compat","model":"anthropic/claude-sonnet-5.5","base_url":"https://openrouter.ai/api/v1"}]' \
  parley-sim-env
```

## Scripted vs adaptive

Training or evaluating against one kind of counterpart hides how your agent
behaves against the other. To measure the gap, serve one container per
counterpart type and run the same seeds through each; parley-sim's matched
runner and report (`parley-sim matched`, `parley-sim report`) do the pairing,
noise floors and statistics for you.
