# parley-sim

Test your agent against users that adapt to it, and measure how much that
changes the result.

Most agent evaluations put the agent against a scripted user: a fixed policy,
a decision tree, a replayed transcript. That user never responds to what the
agent does. parley-sim runs the same scenarios twice, once against a scripted
counterpart and once against an adaptive LLM counterpart, and tells you
whether the difference is real, noise, or too close to call.

- **Matched pairs.** Every scenario is played with the same seed against each
  kind of counterpart, so a difference comes from the counterpart, not the
  scenario.
- **A measured noise floor.** Each adaptive cell is replayed with an identical
  setup. A difference only counts when it is larger than that replay-to-replay
  variation.
- **Equivalence, not just difference.** Give a margin and get a verdict per
  metric: equivalent, different or inconclusive.
- **Judge-free metrics.** Outcomes are computed from the environment's state
  and exact ground truth, not scored by an LLM judge.
- **Replayable traces.** Every decision, scripted or model, accepted or
  declined, is logged and can be replayed through the environment.
- **Bring your own environment.** Implement a small protocol and the runner,
  report, adapters and traces work unchanged.
- **Runs where you train.** Adapters for [NeMo Gym](docs/adapters.md#nemo-gym)
  and [OpenEnv](docs/adapters.md#openenv), including a Docker package.

Status: pre-release (0.0.1). The API may change before 0.1.

## What we found with it

In a pre-registered run on the bundled support-desk environment (Claude Opus
5.5 and Sonnet 5.5 as agents, Claude customers, 40 scenarios each, 400
episodes):

- A plain LLM customer and an LLM customer conditioned on a sampled
  [NeMo UserSim](https://github.com/NVIDIA-NeMo/UserSim) persona produced
  **equivalent** results for the agent on the two pre-registered outcomes,
  surplus share and negotiation length, within margins set in advance.
  (Surplus share is the agent's gain as a fraction of the most it could have
  gained.)
- Persona customers behaved more variedly: different opening offers and more
  varied wording.
- Against scripted customers the agents gained about 6 to 13 points less of
  what they could have (depending on the replicate and agent), and
  negotiations ran longer, than against plain LLM customers. Persona
  customers pointed the same way, but that gap was no larger than
  replay-to-replay variation, so the pre-registered hypothesis that both LLM
  arms differ from scripted was **not supported**.

This is one small environment with one model family on both sides; read it as
a worked example of the method. Full numbers, configs and reports are in
[`results/`](results/README.md).

## Install

parley-sim is not on PyPI yet. From a checkout, with [uv](https://docs.astral.sh/uv/):

```bash
git clone https://github.com/Parley-Labs/parley-sim && cd parley-sim
uv sync --extra openai --extra anthropic   # pick the extras you need, in one command
```

`uv sync` installs exactly the extras you name and removes the others, so list
them all in one command:

| Extra | For |
| --- | --- |
| `openai` | OpenAI-compatible APIs (OpenAI, OpenRouter, vLLM, NIM) |
| `anthropic` | The Anthropic API |
| `usersim` | UserSim persona counterparts |
| `openenv` | The OpenEnv adapter |
| `gym` | The NeMo Gym adapter (Python 3.13.14 or newer; cannot be combined with `anthropic`) |

`pip install -e ".[openai,anthropic]"` works too. Python 3.11 or newer.

## Quickstart

Run the whole pipeline offline, with no API keys. Scripted policies play
every seat, so the "adaptive" arm here is not adaptive; it only checks that
everything runs:

```bash
uv run parley-sim smoke
```

It prints a report and `replay check: ok`.

Then a real comparison. Describe the agents and counterparts in a config:

```yaml
run_id: my-first-run
env: support_desk
seeds: 1000:1020                 # 20 scenarios
lineup:
- name: agent-model
  provider: openai_compat
  model: anthropic/claude-sonnet-5.5
  base_url: https://openrouter.ai/api/v1
- name: customer-model
  provider: openai_compat
  model: anthropic/claude-sonnet-5.5
  base_url: https://openrouter.ai/api/v1
focal_models: [agent-model]                    # the agent under test
scripted_counterparts: [hardliner, reasonable, conceder]
adaptive_counterparts: [customer-model]
max_total_tokens: 1000000                      # token cap, checked as each episode starts
```

Run it, then report:

```bash
export OPENROUTER_API_KEY=...
uv run parley-sim matched my-run.yaml --out runs/my-first-run
uv run parley-sim report runs/my-first-run --margins focal_surplus_share=0.08,turns=0.8
```

Re-running `matched` with the same config resumes where it stopped. The
[configuration reference](docs/configuration.md) covers providers, extra arms,
budgets and the refusal breaker.

## How to read a report

For each agent you get:

1. **Means per arm** for every metric, with its kind (what the metric is
   about; see [concepts](docs/concepts.md#metric-kinds)).
2. **Paired comparisons** between every two arms: the mean difference, 95%
   and 90% intervals, the noise floor, and a verdict.
3. **Diversity**: how much each arm's results and counterpart behavior vary
   across scenarios, and how varied the counterpart's opening messages are.
4. **Acceptance by offer quality**: the agent's decisions facing an offer it
   was authorised to accept, accepted or declined, binned by the offer's
   quality. A standing offer is counted at each of the agent's decisions
   while it stands.

[Concepts](docs/concepts.md) explains each column and the verdict rules.

## Documentation

| Doc | What's in it |
| --- | --- |
| [Concepts](docs/concepts.md) | Arms, matched pairs, noise floor, verdicts, metric kinds, diversity |
| [Configuration](docs/configuration.md) | Every config field, providers, budgets, safety, resuming |
| [Your own environment](docs/custom-environments.md) | The environment protocol and how to plug one in |
| [Adapters](docs/adapters.md) | Running episodes inside NeMo Gym and OpenEnv |
| [Persona counterparts](docs/personas.md) | Using NeMo UserSim personas as a counterpart arm |
| [Claim metrics](docs/claims.md) | The experimental honesty metric and how to validate it |

## The bundled environment

`support_desk` is a retention negotiation. A customer wants to cancel; the
agent can offer a discount, a service credit, a plan upgrade and billing
terms; the discount and the credit are capped by private authorization
limits. The customer has private preferences and a walk-away value. Every
possible deal is enumerable, so the efficient frontier and each side's best
outcome are exact. It is deliberately small and easy to read; it exists to
show the method, and to be replaced by yours.

## Command line

| Command | Does |
| --- | --- |
| `parley-sim smoke` | Offline end-to-end check, no keys |
| `parley-sim matched CONFIG --out DIR` | Run (or resume) a matched grid; exits 1 if any cell errored, 2 if the run halted |
| `parley-sim report DIR... [--margins k=v,...] [--json] [--env NAME]` | Paired report across one or more run directories |
| `parley-sim verify TRACE... [--env NAME]` | Replay traces through the environment and check every logged field |
| `parley-sim claims DIR --prefix P [--env NAME]` | Export a blind sample of agent claims for labelling |

`--env` is needed only for an environment loaded by import path
(`my_pkg.envs:ENTRY`): parley-sim never imports a module named inside a
trace or run directory, so a downloaded trace can't run code.
| `parley-sim kappa BLIND KEY` | Agreement between the claim rules and human labels |

## License

Apache-2.0. Persona datasets used by the optional UserSim arm carry their own
licence; see [persona counterparts](docs/personas.md#licence).
