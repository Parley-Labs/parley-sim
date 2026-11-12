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

The protocol is in `parley_sim/core/protocol.py`. The rules that matter:

- **Pure and sequential.** Exactly one seat acts at a time (`to_act`), and
  `apply(state, seat, action)` returns a new state without changing the old
  one. Keep the scenario inside the state rather than on `self`, so any state
  can be replayed.
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

`render(observation) -> (system, user)` builds the prompt; optionally,
`render_update(previous, observation, submitted) -> str` builds the next user
message of a conversation seat: what changed since its previous turn,
including its own move whenever the recorded move differs from `submitted`,
the action it last sent (`None` when its last turn produced no usable reply);
see
`conversation` in [configuration](configuration.md#model-specs)), and without
it a conversation seat resends the full rendered prompt each turn;
`action_schema` is the JSON schema sent to providers for structured output;
`parse(payload) -> (action, explanation)` turns the model's reply into an
action, raising `MalformedOutput` when it can't. Bump `prompt_version`
whenever the prompt changes: it is recorded in every trace.

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
   be importable (installed, or on `PYTHONPATH`). To replay its traces, pass
   the same path: `parley-sim verify TRACE --env my_pkg.envs:ENTRY`.
2. **Register in code:** `parley_sim.envs.register("my_env", ENTRY)`, then
   `env: my_env`. Useful in notebooks and tests.
3. **Entry point in your package**, so `env: my_env` works everywhere. Name
   the entry point after your environment's `name` attribute and `verify`
   needs no flag either:

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
