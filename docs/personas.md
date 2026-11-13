# Persona counterparts

parley-sim can condition a model counterpart on a persona from NVIDIA's
[NeMo UserSim](https://github.com/NVIDIA-NeMo/UserSim): a sampled person from
the Nemotron-Personas datasets, with an OCEAN personality profile, a
disclosure style and an interaction style. Use it as an extra arm to see
whether personas change your agent's results, or to widen the range of user
behavior you test against.

## How it works

The persona block, behavioral profile and styles come from UserSim's own
functions, derived and ordered the way UserSim builds its episodes. parley-sim
then adds the environment's role, rules, private information and move format.
UserSim's users are help-seekers judged by UserSim's own judges, which have no
notion of binding offers, so the negotiation half is parley-sim's. Label
results "UserSim personas and behavior model", not "UserSim".

Each seed gets one persona, the same in every arm, cell and model, so a
persona arm stays a matched comparison. The trace records the persona's
derived fields (OCEAN scores and labels, styles, UserSim version) with every
decision.

## Setup

```bash
uv sync --extra usersim --extra openai   # name every extra you use, in one command
export NEMO_TELEMETRY_ENABLED=false
uv run usersim setup-ngc          # then configure the NGC CLI with your NGC key
```

`uv sync` removes any extra you don't name, so add the provider extra your
lineup uses (`openai` for the config below) to the same command.

Download the en_US Nemotron-Personas dataset from NGC as UserSim's
[persona docs](https://github.com/NVIDIA-NeMo/UserSim/blob/main/docs/personas.md)
describe. The NGC copy carries the OCEAN traits the behavior model needs. Then
sample a panel (no model calls):

```bash
uv run usersim panel --locale en_US --num-personas 200 \
  --out ~/.cache/parley-sim/usersim-en_US.parquet
```

Check the panel has OCEAN data: if every persona's interaction style comes out
"neutral", the traits are missing and the arm would not be persona-driven.

## Config

Add a lineup entry with `persona: usersim`, put it in an extra arm, and point
`persona_file` at the panel:

```yaml
lineup:
- name: sonnet-usersim
  provider: openai_compat
  model: anthropic/claude-sonnet-5.5
  base_url: https://openrouter.ai/api/v1
  persona: usersim
extra_arms:
  usersim: [sonnet-usersim]
persona_file: ~/.cache/parley-sim/usersim-en_US.parquet
```

`configs/equiv-sd3.yaml` is a full example, alongside plain LLM and scripted
arms on the same seeds. Only `en_US` personas are supported so far.

Each seed's persona is drawn from the panel, so the run records a digest of
the panel's contents. Resuming into a directory built from another panel is
refused, and so is a report that joins one persona arm (or a persona-conditioned
focal model) across runs whose panels differ or weren't recorded.
Differently named arms may use different panels; the panel is then part of
what differs between them.

Persona prompts are longer, so expect more input tokens per episode: in our
runs, about 19.6k against 11.3k for the same customer without a persona.

## Licence

The persona datasets carry their own licence, separate from UserSim's and
parley-sim's Apache-2.0. Per UserSim's documentation, the NGC datasets are
under the NVIDIA Dataset License Agreement, which permits use for training
and developing AI solutions and adds an AI ethics clause; the Hugging Face
collection is CC BY 4.0. Redistribution of the NGC data is not clearly
covered. Read the agreement yourself before you share anything derived from
it.

What parley-sim does about it:

- the panel file stays outside the repository (`*.parquet` is git-ignored);
- the published [results](../results/README.md) include the persona arm's
  reports but not its traces, because those carry each persona's trait scores
  and in-character messages. Rebuild a panel and re-run the config to
  reproduce them.
