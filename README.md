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
5.5 and Sonnet 5.5 as agents, 40 scenarios each, 400 episodes):

- A plain LLM customer and an LLM customer driven by a mined
  [NeMo UserSim](https://github.com/NVIDIA-NeMo/UserSim) persona produced
  **equivalent** outcomes for the agent, within margins set in advance.
- Persona customers behaved more variedly: different opening offers and more
  varied wording.
- Scripted customers did not stand in for either. Against them the agents
  kept about 10 points less of each deal and negotiations ran longer.

Full numbers, configs and reports are in [`results/`](results/README.md).

## Install

parley-sim is not on PyPI yet. From a checkout, with [uv](https://docs.astral.sh/uv/):

```bash
git clone https://github.com/Parley-Labs/parley-sim && cd parley-sim
uv sync                         # core: runner, report, support_desk env
uv sync --extra openai          # OpenAI-compatible APIs (OpenAI, OpenRouter, vLLM, NIM)
uv sync --extra anthropic       # Anthropic API
uv sync --extra usersim         # UserSim persona counterparts
uv sync --extra gym             # NeMo Gym adapter
uv sync --extra openenv         # OpenEnv adapter
```

`pip install -e ".[openai]"` works the same way. Python 3.11 or newer.

## Quickstart

Run the whole pipeline offline, with no API keys. Scripted policies play
every seat, so the "adaptive" arm here is not adaptive; it only checks that
everything runs:

```bash
uv run parley-sim smoke
```

It ends with a report and `replay check: ok`.

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
max_total_tokens: 1000000                      # hard spend cap
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
4. **Acceptance by offer quality**: every offer the agent faced, accepted or
   declined, binned by how good it was.

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
terms, each within private authorization limits. The customer has private
preferences and a walk-away value. Every possible deal is enumerable, so the
efficient frontier and each side's best outcome are exact. It is deliberately
small and easy to read; it exists to show the method, and to be replaced by
yours.

## Command line

| Command | Does |
| --- | --- |
| `parley-sim smoke` | Offline end-to-end check, no keys |
| `parley-sim matched CONFIG --out DIR` | Run (or resume) a matched grid |
| `parley-sim report DIR... [--margins k=v,...] [--json]` | Paired report across one or more run directories |
| `parley-sim verify TRACE... [--env NAME]` | Replay traces through the environment |
| `parley-sim claims DIR --prefix P` | Export a blind sample of agent claims for labelling |
| `parley-sim kappa BLIND KEY` | Agreement between the claim rules and human labels |

## License

Apache-2.0. Persona datasets used by the optional UserSim arm carry their own
licence; see [persona counterparts](docs/personas.md#licence).
