# Configuration

A run is one YAML file passed to `parley-sim matched CONFIG --out DIR`. The
bundled `configs/` folder has working examples: `pilot.yaml` (two arms),
`equiv-sd3.yaml` (three arms, pre-registered) and `family-sd4.yaml` (extra
arms run into their own directory).

## Run fields

| Field | Default | Meaning |
| --- | --- | --- |
| `run_id` | required | Name for the run; prefixes every cell key |
| `env` | `support_desk` | Environment name, or an import path such as `my_pkg.envs:ENTRY` (see [your own environment](custom-environments.md)) |
| `env_config` | `{}` | Options passed to the environment, e.g. `{max_turns: 10}` for `support_desk` |
| `seeds` | required | Half-open range `start:stop`, e.g. `1000:1040` for 40 scenarios |
| `focal_seat` | the env's default | Which seat is under test (`agent` in `support_desk`) |
| `lineup` | required | Model specs, below |
| `focal_models` | required | Lineup names to evaluate in the focal seat; each gets its own grid |
| `scripted_counterparts` | required | Policy names for the `soc` arm |
| `adaptive_counterparts` | required | Lineup names for the `aoc` arm |
| `extra_arms` | `{}` | More model-counterpart arms, `{arm_name: [lineup names]}` |
| `run_arms` | all | Only run these arms; the rest can live in another directory |
| `persona_file` | none | Persona panel for lineup entries with `persona: usersim` ([personas](personas.md)) |
| `self_null_replicates` | `1` | Identical replays of each model-counterpart cell, for the noise floor; `0` skips them |
| `concurrency` | `4` | Episodes in flight at once |
| `max_total_tokens` | none | Token cap (input + output) across the run, checked as each episode starts; see below |
| `max_refusals` | `3` | Halt the whole run after this many provider refusals |

The grid size is seeds × focal models × (1 for `soc` + model arms × (1 +
`self_null_replicates`)), counting only the arms in `run_arms` when it is set.
`family-sd4.yaml`, for example, runs only its 2 model arms (`run_arms`), with
40 seeds, 2 focal models and no replicates: 40 × 2 × 2 = 160 episodes.

## Model specs

Each `lineup` entry:

| Field | Default | Meaning |
| --- | --- | --- |
| `name` | required | How the config refers to it |
| `provider` | required | `anthropic`, `openai`, `openai_compat`, `scripted` or `harness` |
| `model` | required | The provider's model ID (for `scripted`, a policy name) |
| `base_url` | none | For `openai_compat`: the gateway, e.g. `https://openrouter.ai/api/v1` |
| `api_key_env` | per provider | Environment variable holding the key |
| `temperature` | omitted | Sent only if set; not accepted for `anthropic` |
| `effort` | omitted | Reasoning effort: `low`, `medium`, `high`, `xhigh`, `max` |
| `max_tokens` | `4096` | Output cap per call |
| `timeout_s` | `120` | Per-call timeout |
| `provider_params` | none | Merged into the request last, e.g. OpenRouter routing |
| `persona` | none | `usersim` conditions this seat on a UserSim persona |
| `conversation` | `false` | Play the seat as one growing conversation: after the first turn, each prompt is only what changed since the seat last acted, sent after its own earlier replies. Off, every turn is a freshly rendered prompt. The NeMo Gym adapter uses it for the trained seat; turn it on locally to evaluate with the same prompt shape |

### Providers

| Provider | Key variable | Notes |
| --- | --- | --- |
| `anthropic` | `ANTHROPIC_API_KEY` | Structured outputs; adaptive thinking with summaries recorded in the trace; `effort` goes to `output_config.effort` |
| `openai` | `OPENAI_API_KEY` | `response_format` JSON schema; `effort` goes to `reasoning_effort` |
| `openai_compat` | `OPENROUTER_API_KEY` | Any OpenAI-compatible gateway (OpenRouter, vLLM, NIM); `effort` goes to OpenRouter's `reasoning.effort` |
| `scripted` | none | Plays a named policy through the text interface; for offline tests |
| `harness` | none | Calls are made by an external harness (used by the NeMo Gym adapter) |

The key is read from `api_key_env` if set, and a key is only ever sent to the
host it belongs to, over HTTPS (plain `http://` is allowed only to localhost,
for a local server such as vLLM):

- `openai_compat` requires `base_url` and never falls back to
  `OPENAI_API_KEY`;
- `openai` with a custom `base_url` requires an explicit `api_key_env`, so
  `OPENAI_API_KEY` is never sent to another host;
- `openai` and `anthropic` also treat `OPENAI_BASE_URL` and
  `ANTHROPIC_BASE_URL`, which their SDKs read, as custom endpoints;
- `anthropic` with a custom `base_url` requires an explicit `api_key_env`,
  and if `api_key_env` names a variable that
  is unset, it fails instead of falling back to `ANTHROPIC_API_KEY`.

To pin OpenRouter to one upstream and get cost reporting (cost is reported
only for calls OpenRouter bills; calls made with your own provider key
through OpenRouter show $0):

```yaml
provider_params:
  extra_body:
    provider: {order: [Anthropic], allow_fallbacks: false}
    usage: {include: true}
```

For a model that might not support structured outputs everywhere, use
`provider: {require_parameters: true}` instead of pinning.

## Safety and spend

- **Refusal breaker.** All seats in a run share a breaker. It trips on any
  401 or 403, or after `max_refusals` provider refusals. Once tripped, no
  further calls are made, the remaining cells are recorded as halted, and
  `matched` exits with code 2 and a `RUN HALTED` message. Repeated refusals
  can get a provider account suspended; find the cause before re-running.
- **Token cap.** With `max_total_tokens`, cells that would start after the
  cap is reached are recorded as `skipped_budget`. Episodes already running
  finish, so a run can overshoot by up to `concurrency` episodes. It caps
  tokens, not money.
- **Refusals are strikes, not substitutions.** No client silently falls back
  to another model, which would contaminate the evaluation.

## Resuming

Run the same command again. A cell is skipped only if the manifest says it is
done and its trace has a footer; anything else is run again. parley-sim
refuses to resume into a directory created from a different config, so
changing the experiment means a new `--out`.

## Output directory

| Path | Contents |
| --- | --- |
| `manifest.jsonl` | The config, its hash, and one row per cell with status, trace path and usage |
| `traces/<cell_key>.jsonl` | One trace per episode ([format](concepts.md#traces)) |

Produce the report with `parley-sim report DIR > DIR/report.md`, or
`--json` for the full structure.
