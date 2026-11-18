# Your own environment

parley-sim is built to wrap your environment. Once you give it one
`EnvEntry`, the matched runner, report, trace replay and both adapters work
unchanged. [`examples/haggle.py`](../examples/haggle.py) is a complete, tested
example in about 200 lines: a seller and a buyer haggling over one price. Run
it offline:

```bash
PYTHONPATH=examples uv run parley-sim matched examples/haggle.yaml --out runs/haggle
```

## What you provide

An `EnvEntry` (from `parley_sim.envs`) bundles:

| Field | What it is |
| --- | --- |
| `env` | A zero-argument callable returning your environment (usually the class) |
| `make_policy` | `(name, seat) -> Policy`: your scripted counterparts |
| `text` | A zero-argument callable returning the text interface: how a model plays a seat |
| `focal_seat` | The seat under test by default |
| `episode_metrics` | `(trace_lines) -> {metric: float or None}` |
| `metric_kinds` | `{metric: kind}`, one of intrinsic, relational, covariate, counterpart |
| `decision_rows` | `(trace_lines) -> rows` for the acceptance table; return `[]` to skip it |

Two optional hooks unlock more of the report: `counterpart_opening` (the
counterpart's first message, for wording diversity) and `claim_rows` (for
[claim validation](claims.md)).

## The environment

The protocol is `Environment` in `parley_sim/core/protocol.py`. An
environment has two attributes, `name` and `version`, and these methods:

| Method | Returns |
| --- | --- |
| `generate(seed, config)` | The scenario (instance) for a seed |
| `instance_to_json(instance)` | The instance as JSON, recorded in the trace header |
| `ground_truth(instance)` | Exact quantities your metrics normalise by |
| `seats(instance)` | The seat names, in order |
| `initial_state(instance)` | The starting state |
| `to_act(state)` | The seat that acts next, or `None` once the episode is over |
| `turn(state)` | The current turn number |
| `observe(state, seat)` | What that seat sees |
| `decision_context(state, seat)` | What the decision is answering |
| `apply(state, seat, action)` | `(new_state, events)` |
| `fallback_action(state, seat)` | The move applied after an illegal one |
| `outcome(state)` | `{seat: value}` at the end of the episode |
| `state_to_json(state)` | The state as JSON, hashed into the trace |

The rules that matter:

- **Pure and sequential.** Exactly one seat acts at a time (`to_act`), and
  `apply(state, seat, action)` returns `(new_state, events)` without changing
  the old state; `events` is a list of JSON objects logged in the trace. Keep
  the scenario inside the state rather than on `self`, so any state can be
  replayed.
- **Seeded.** `generate(seed, config)` must return the same scenario for the
  same seed. Matched pairs and `verify` depend on it.
- **JSON in, JSON out.** Actions and observations are plain JSON objects, and
  `state_to_json`, `instance_to_json` and `ground_truth` must be
  JSON-serialisable. This is what lets traces, model I/O and the adapters work
  without per-environment code.
- **`observe` is the only view of state** a seat ever gets, so private
  information stays private.
- **Illegal moves raise `IllegalAction`.** The runtime records a strike and
  applies your `fallback_action` instead. Make the fallback neutral (in the
  example, repeating the seat's last offer), so a model's formatting mistake
  doesn't hand the other side a win.
- **`decision_context`** says what a decision is answering (e.g. the standing
  offer). It is logged with every decision, so declines are as visible as
  acceptances.
- **`ground_truth`** should hold whatever exact quantities your metrics
  normalise by: the best each side could get, the maximum joint surplus.
  Computed ground truth is what keeps the metrics judge-free.

## Scripted counterparts

A policy has a `name`, a `params` dict (recorded in the trace header) and
`decide(observation) -> action`. It sees exactly what a model in that seat
would see. Give your policies a range (in the example, `firm` and
`flexible`), because the scripted arm rotates through them across seeds.

## The text interface

A text interface has:

- `render(observation) -> (system, user)`, which builds the prompt;
- `action_schema`, the JSON schema sent to providers for structured output;
- `parse(payload) -> (action, explanation)`, which turns the model's reply
  into an action, raising `MalformedOutput` when it can't;
- `prompt_version`. Bump it whenever the prompt changes: it is recorded in
  every trace.

One method is optional. A seat with `conversation` on (see
[configuration](configuration.md#model-specs)) plays one growing
conversation. For it, `render_update(previous, observation, submitted) -> str`
builds each user message after the first: what changed since the seat's
previous turn. `submitted` is the action the seat last sent, or `None` when
its last turn produced no usable reply; restate the seat's own move whenever
the recorded move differs from it. Without `render_update`, a conversation
seat gets the full rendered prompt each turn.

Don't ask the model to write out its private reasoning in a schema field.
Some providers treat that as an attempt to extract reasoning and refuse.
Reasoning summaries, where a provider offers them, are recorded separately.

## Metrics

`episode_metrics` reads a trace: `lines[0]` is the header (spec, instance,
ground truth), `lines[-1]` the footer (status, outcomes, final state), and
the lines in between are decisions and events. Return `None` for every metric
when the footer status is not `complete`. Name a metric
`focal_surplus_share`, `turns` or `cp_first_offer_share` to get its spread in
the diversity table, and use `focal_surplus_share` (or set the adapter's
`reward_metric`) for the adapters' reward.

Declare each metric's kind honestly: it is how a reader tells "the agent
behaved differently" from "the counterpart behaved differently".

## Plugging it in

Three ways, from quickest to most permanent:

1. **Import path in the config:** `env: my_pkg.envs:ENTRY`. The module must
   be importable (installed, or on `PYTHONPATH`). parley-sim imports a path
   only when you give it, never when it is read from a trace or manifest,
   so to report on or replay such a run in a new process pass the same path:
   `parley-sim report DIR --env my_pkg.envs:ENTRY`,
   `parley-sim verify TRACE --env my_pkg.envs:ENTRY`. An import path can't
   reuse the name of a built-in or registered environment.
2. **Register in code:** `parley_sim.envs.register("my_env", ENTRY)`, then
   `env: my_env`. Useful in notebooks and tests.
3. **Entry point in your package**, so `env: my_env` works everywhere. Name
   the entry point after your environment's `name` attribute and `report`
   and `verify` need no flag either:

   ```toml
   [project.entry-points."parley_sim.envs"]
   my_env = "my_pkg.envs:ENTRY"
   ```

`ENTRY` may also be a zero-argument function that returns an `EnvEntry`.

## Checklist

- [ ] The same seed gives the same scenario.
- [ ] `parley-sim verify` passes on a few traces.
- [ ] A scripted-only run (like `examples/haggle.yaml`) produces a sensible
      report before you spend on models.
- [ ] Each scripted policy reaches deals sometimes; a counterpart that
      always walks away makes every comparison trivial.
- [ ] The fallback move is neutral.
