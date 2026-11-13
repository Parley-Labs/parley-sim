# parley-sim for OpenEnv

Negotiation environments from [parley-sim](../../../README.md). Your model plays
the **agent**; the environment plays the **counterpart**: a scripted persona or
another LLM. Every episode has exact ground truth (all possible deals are
enumerable), so rewards are computed, not judged.

The demo environment, `support_desk`, is a retention negotiation. A customer
wants to cancel; your model is the support agent with private authorization
limits, negotiating discount, credit, upgrade and billing term.

## Run it

Locally, from this folder (`integrations/openenv/parley_sim_env`: `--project .`
means the current directory, so run it here):

```bash
uv run --project . server            # http://localhost:8000, bound to 127.0.0.1
```

With Docker, from the parley-sim repository root:

```bash
docker build -f integrations/openenv/parley_sim_env/server/Dockerfile -t parley-sim-env .
docker run -p 127.0.0.1:8000:8000 parley-sim-env
```

The server has no authentication. `-p 127.0.0.1:8000:8000` publishes the port
on this machine only; without the `127.0.0.1:` prefix, Docker publishes it on
every interface.

If you deploy with `openenv push`, pass `--no-interface`. By default `openenv
push` turns on OpenEnv's web interface, which would share one environment
between all its users; this server never mounts it (it logs a warning and
ignores `ENABLE_WEB_INTERFACE`), so without the flag the Space's page points
at a `/web` path that does not exist.

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
parley-sim's own runs. A reply may be up to 65,536 characters. Once the episode
is done, `env.state()` also carries the full episode trace, including the
counterpart's moves, unless the server runs with `PARLEY_RELEASE_TRACE=never`
(see below).

Play goes through OpenEnv's WebSocket sessions, which `ParleySimEnv` uses. The
server serves only `/ws`, `/health`, `/schema`, `/metadata` and the API docs.
The stateless `/reset`, `/step`, `/state` and `/mcp` routes answer 405, and
OpenEnv's other routes are dropped: each such call would build a fresh
environment outside the session limit, idle timeout and refusal breaker. `ParleySimEnv` waits up to 300 seconds for each reply (OpenEnv's
default is 60), because one LLM counterpart turn can take two attempts of its
`timeout_s` (120 seconds by default). If you raise `timeout_s`, seat more
than one model counterpart, or a counterpart can act several times before your
turn, raise it too:
`ParleySimEnv(base_url=..., message_timeout_s=600)`.

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
| `PARLEY_SESSION_TIMEOUT` | `600` | Close sessions idle this many seconds. Must cover a whole reset or step. The server refuses to start below twice the sum of the seated model counterparts' `timeout_s`; allow more if a seat can act several times per turn. |
| `PARLEY_SEED_SECRET` | none | Keys requested seeds into scenario seeds, so a client can't compute a scenario from its seed. Use with `PARLEY_RELEASE_TRACE=never`. |
| `PARLEY_RELEASE_TRACE` | `after_done` | `after_done`: `env.state()` carries the full trace once an episode is done. `never`: it only ever shows what your model's seat saw, plus the outcomes and metrics. Any other value fails startup. |

The server has no authentication, and with model counterparts every session
spends your provider keys. `uv run --project . server` binds to localhost; the
Docker image listens on all interfaces inside the container, so publish its
port as `-p 127.0.0.1:8000:8000`, and expose it beyond this machine only on a
trusted network.

During an episode, `env.state()` shows only what your model's seat can see.
Once an episode ends, its full trace, with the scenario, ground truth and the
counterpart's decisions, is released. The client chooses the seeds, so it can
end an episode at once, read that trace, and replay the same seed with the
counterpart's private values in hand. With `PARLEY_RELEASE_TRACE=never`,
`state()` never releases the trace, so counterpart decisions stay in the
server. That does not make the scenario secret: the generator is public and
deterministic, so a client that picks a seed can compute the counterpart's
private values itself. The model being trained only ever sees what your
client passes it.

Scripted customers: `policy:hardliner`, `policy:reasonable`, `policy:conceder`.
An LLM customer needs a lineup entry and its provider key at runtime:

```bash
docker run -p 127.0.0.1:8000:8000 -e OPENROUTER_API_KEY \
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
