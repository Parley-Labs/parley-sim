# Results

Two runs on the bundled `support_desk` environment, with Claude Opus 5.5 and
Claude Sonnet 5.5 as the agents under test, called through OpenRouter and
pinned to Anthropic, at low reasoning effort.

| Run | Config | Episodes | What it asks | Status |
| --- | --- | --- | --- | --- |
| [`equiv-sd3`](equiv-sd3/report.md) | [`configs/equiv-sd3.yaml`](../configs/equiv-sd3.yaml) | 400 | Do persona customers change the agent's outcomes, compared with plain LLM and scripted customers? | Pre-registered |
| [`family-sd4`](family-sd4/joint-report.md) | [`configs/family-sd4.yaml`](../configs/family-sd4.yaml) | 160 | Do customers from other model families (Muse, Nemotron) behave as variedly as persona customers? | Exploratory |

Both use seeds 1060 to 1099, so `family-sd4`'s arms join `equiv-sd3`'s in
[`joint-report.md`](family-sd4/joint-report.md).

## equiv-sd3 (pre-registered)

The analysis plan is the header of its config, committed before any episode
existed. Three customer arms: scripted (`soc`), Claude with a NeMo UserSim
persona (`usersim`), and plain Claude (`aoc`). Margins: 0.08 surplus share,
0.8 turns.

| Hypothesis | Result |
| --- | --- |
| H1: plain and persona customers give equivalent agent outcomes | Supported for both agents. Surplus share differed by +0.008 (Opus) and +0.022 (Sonnet), 90% intervals inside ±0.08; turns inside ±0.8 |
| H2: persona customers behave more variedly | Supported for both agents. First-offer spread 0.26 to 0.29 against 0 for plain customers; opening-message wording more varied (permutation p < 0.001) |
| H3: both LLM arms differ from scripted on surplus share | Partly. Plain LLM vs scripted is different (+0.11 Opus, +0.10 Sonnet). Persona vs scripted (+0.105, +0.080) excludes zero but sits just under the noise floor, so it is inconclusive |

Every episode in every arm ended in a deal. Spend: about 5.5M input and 0.47M
output tokens.

## family-sd4 (exploratory)

Muse (`meta/muse-wonder`) and Nemotron (`nvidia/nemotron-3-ultra-550b-a55b`)
as plain customers, same prompt, no persona, no replicates.

- Their first offers spread almost as widely as persona customers' (0.20 to
  0.24), where plain Claude customers never varied.
- Their opening wording was no more varied than plain Claude's; persona
  customers stayed clearly ahead (p ≤ 0.001).
- The customer model moved the agent's result: against Muse, agents kept 55%
  (Opus) and 62% (Sonnet) of the deal, against 68% and 71% with plain Claude
  customers. Muse produced an invalid move on 26% of its turns (Nemotron on
  none), which may explain part of that.

Spend: $7.60 as reported by OpenRouter.

## What is and isn't here

| Included | Withheld |
| --- | --- |
| Reports as published (`report.md`, `report.json`, `joint-report.md`) | Traces of the `usersim` arm (160 episodes) |
| Manifests, with the `usersim` rows removed | The persona panel itself |
| Traces of every other arm (400 episodes) | |
| Claim-validation samples and human labels, rounds 1 and 2 | |

The `usersim` traces record each persona's trait scores from NVIDIA's persona
dataset, and the customers' in-character messages. The dataset's licence
covers use for developing AI, not obviously redistribution, so we leave them
out. Their results are in the published reports; to reproduce them, build a
panel ([personas](../docs/personas.md)) and re-run the config.

`scripts/curate_results.py` made this folder from the full run directories.

## Recomputing

```bash
uv run parley-sim report results/equiv-sd3 --margins focal_surplus_share=0.08,turns=0.8
uv run parley-sim verify results/family-sd4/traces/*.jsonl
```

The recomputed `equiv-sd3` report matches the published one for every
comparison that doesn't involve the withheld arm, except the two experimental
claim metrics (`limit_claim`, `understated_limit`). Those changed because the
claim rules were fixed after the run ([claim metrics](../docs/claims.md)).
