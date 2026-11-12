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

**The main metric.** `focal_surplus_share` is the agent's surplus (its value
for the deal minus its value for no deal) divided by the most surplus it could
have achieved in a deal both sides would accept. 0.68 means the agent got 68%
of what it could have. It can exceed 1 when the counterpart accepts a deal
below its own walk-away value (one episode in these runs).

## equiv-sd3 (pre-registered)

The analysis plan is the header of its config, committed before any episode
existed. Three customer arms: scripted (`soc`), Claude with a NeMo UserSim
persona (`usersim`), and plain Claude (`aoc`). Margins: 0.08 surplus share,
0.8 turns.

| Hypothesis | Result |
| --- | --- |
| H1: plain and persona customers give equivalent agent outcomes | **Supported** for both agents. Surplus share differed by +0.008 (Opus) and +0.022 (Sonnet), 90% intervals inside ±0.08; turns inside ±0.8 |
| H2: persona customers behave more variedly | **Supported** for both agents. First-offer spread 0.26 to 0.29 against 0 for plain customers; opening-message wording more varied (permutation p ≈ 0.001 or below) |
| H3: both LLM arms differ from scripted on surplus share | **Not supported as pre-registered.** Plain LLM vs scripted is different (+0.11 Opus, +0.10 Sonnet). Persona vs scripted (+0.105, +0.080) has an interval excluding zero but sits just under its noise floor (0.106, 0.082), so the rule calls it inconclusive |

Every episode in every arm ended in a deal. About 5.5M input and 0.47M output
tokens. The Claude calls were billed to our own Anthropic key through
OpenRouter, so the reports show $0 for them.

Three caveats a reader should weigh:

- **The noise floor is conservative.** It is the mean per-episode gap between
  two replays of the same cell, not a standard error, so it doesn't shrink
  with more seeds. H3's persona comparison misses it by 0.001 to 0.002, and
  the floor itself has a bootstrap interval of about ±0.03.
- **Sonnet's plain-LLM "different" is as fragile as the persona
  "inconclusive".** The verdicts use replicate r0, as pre-registered. On the
  identical replicate r1, Sonnet's plain LLM vs scripted difference is +0.064,
  95% interval (0.012, 0.121), below its noise floor (0.093), so the rule
  would not call it different. Opus is different on both replicates (+0.113,
  +0.128).
- **The margin was set from the pilot.** 0.08 was half the smallest
  scripted-vs-LLM gap in the pilot (0.16). In this run that gap was smaller
  (0.10 to 0.11), so 0.08 is about three quarters of it. H1's intervals still
  sit well inside it for Opus (upper bound 0.041) and inside it for Sonnet
  (0.065).

## family-sd4 (exploratory)

Muse (`meta/muse-wonder`) and Nemotron (`nvidia/nemotron-3-ultra-550b-a55b`)
as plain customers, same prompt, no persona, no replicates.

- Their first offers spread almost as widely as persona customers' (0.20 to
  0.24), where plain Claude customers never varied.
- Their opening wording was no more varied than plain Claude's; persona
  customers stayed clearly ahead (p ≈ 0.001 or below).
- The customer model moved the agent's result: against Muse, agents' surplus
  share was 0.55 (Opus) and 0.62 (Sonnet), against 0.68 and 0.71 with plain
  Claude customers. Muse made an illegal move on 84 of its 345 turns (24%),
  plus 5 turns lost to provider errors, which may explain part of that.
  Nemotron had none. Four Sonnet episodes, two against each new customer,
  ended without a deal.

Spend: $7.60 reported by OpenRouter for the customer calls (Muse $6.67,
Nemotron $0.94); the Claude agent calls were billed to our own Anthropic key.

## What is and isn't here

| Included | Withheld |
| --- | --- |
| Reports as published (`report.md`, `report.json`, `joint-report.md`) | Traces of the `usersim` arm (160 episodes) |
| Manifests, with the `usersim` rows removed | The persona panel itself |
| Traces of every other arm (400 episodes) | |
| Claim-validation samples and human labels, rounds 1 and 2 (agent sentences only; 104 of the 300 rows come from `usersim`-arm episodes, whose traces are withheld) | |

The `usersim` traces record each persona's trait scores from NVIDIA's persona
dataset, and the customers' in-character messages. The dataset's licence
covers use for developing AI, but redistribution is not clearly covered, so
we leave them out. Their results are in the published reports. Re-running
the config with a freshly built panel reproduces the method, not these exact
numbers, because a new panel draws different personas.

`scripts/curate_results.py` made this folder from the full run directories.

## Recomputing

```bash
uv run parley-sim report results/equiv-sd3 --margins focal_surplus_share=0.08,turns=0.8
uv run parley-sim report results/equiv-sd3 results/family-sd4    # joint-report.md
uv run parley-sim verify results/family-sd4/traces/*.jsonl
```

The published reports were produced by the code at the time of each run.
Recomputing with the current code matches them for every comparison that
doesn't involve the withheld arm, with four known differences:

- the experimental claim metrics (`limit_claim`, `understated_limit`) changed
  because the claim rules were fixed after the run, and are now labelled
  experimental ([claim metrics](../docs/claims.md));
- the acceptance-by-offer-quality tables now leave out offers the agent wasn't
  authorised to accept, which the published tables counted as declines;
- the means table now states how many episodes each arm contributes;
- the `Cells` and `Spend` header lines count only the episodes present, so
  they leave out the withheld arm. `equiv-sd3` recomputes to `Cells: done=240`
  and 2,340,767 input + 273,647 output tokens (published: 400 cells,
  5,477,709 + 465,346); the joint report to `Cells: done=400` and 3,993,323 +
  1,048,026 (published: 560 cells, 7,130,265 + 1,239,725).
