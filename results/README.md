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

A third run, [`harbor-pilot`](#harbor-pilot-pre-registered), repeats the
`equiv-sd3` question with full agent harnesses (Claude Code and Terminus-2)
playing the agent through the [Harbor export](../docs/harbor.md), on fresh
seeds. A fourth, [`harbor-user-study`](#harbor-user-study-pre-registered),
asks whether Harbor's own simulated user can stand in for a parley-sim model
counterpart, on the `meeting_scheduling` environment.

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
| H1: plain and persona customers give equivalent agent outcomes | **Supported** for both agents. The 90% intervals for the plain vs persona difference in surplus share, -0.023 to +0.041 (Opus) and -0.022 to +0.065 (Sonnet), lie inside the ±0.08 margin; turns within ±0.8 (Opus -0.65 to 0.00, Sonnet -0.63 to +0.10) |
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
  (0.10 to 0.11), so 0.08 is about three quarters of it. Read H1 by its
  intervals (upper bounds 0.041 for Opus, 0.065 for Sonnet) more than by the
  word "equivalent".
- **The scripted arm is three bots we wrote,** and the LLM customers are
  Claude, as are the agents. The scripted-vs-LLM gap measures those bots
  against Claude, and depends on how the bots were tuned; the persona vs plain
  comparison is Claude against Claude. `family-sd4` hints that the customer's
  model family moves outcomes more than the persona does, though Muse's
  illegal moves confound that.
- **The pre-registration can't be proven from git.** The plan was committed
  before the run, but the repository's history was rewritten before
  publication, so commit dates no longer show the order. Later runs should be
  registered somewhere external with a timestamp.

### Re-analysis with the margin rule (exploratory)

The noise floor is a single episode's replay wobble, compared against a mean
over 40 seeds, which wobbles about √40 times less: a bar roughly seven times
the noise it stands for. `parley-sim report --verdict-rule margin` averages
each seed's two replays instead, and calls a difference "different" when its
95% interval excludes zero and it is at least the 0.08 margin
([verdict rules](../docs/concepts.md#verdicts)). Re-running
the same episodes through it, with no new runs
([`report-margin-rule.md`](equiv-sd3/report-margin-rule.md)):

| Comparison (surplus share) | Pre-registered rule | Margin rule |
| --- | --- | --- |
| H1, plain vs persona | equivalent (both agents) | equivalent (both agents), narrower intervals |
| H3, Opus plain vs scripted | different (+0.113) | different (+0.121) |
| H3, Opus persona vs scripted | inconclusive (+0.105, floor 0.106) | **different** (+0.117) |
| H3, Sonnet plain vs scripted | different (+0.102) | different (+0.083) |
| H3, Sonnet persona vs scripted | inconclusive (+0.080, floor 0.082) | inconclusive (+0.077, under the 0.08 margin) |

Turns are "different" for every LLM vs scripted comparison under both rules.

Under the margin rule, H3 holds for Opus but not for Sonnet, whose persona gap
sits just under the margin. **This does not replace the pre-registered
result above**, which used the floor rule as committed: the rule was chosen
after seeing the data, so this is a sensitivity check, not a confirmatory
finding. The scripted arm has one replay per seed, so averaging reduces noise
only on the LLM side. A future study should pre-register the margin rule
before running.

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

## harbor-pilot (pre-registered)

The analysis plan is the header of
[`configs/harbor-pilot.yaml`](../configs/harbor-pilot.yaml), committed
before any episode existed. Agents: Harbor's `claude-code` and `terminus-2`,
both on Claude Sonnet 5.5. Customers: scripted (`soc`), plain Claude Sonnet
5.5 (`aoc`), and the same model with a NeMo UserSim persona (`usersim`).
Seeds 1100 to 1139, two replicates for each LLM arm: 200 tasks per agent, 400
episodes, all completed and verified. Verdict rule: margin (replicates
averaged per seed), margins 0.08 surplus share and 0.8 turns. Report:
[`harbor-pilot/report.md`](harbor-pilot/report.md).

| Hypothesis | Agent | Result |
| --- | --- | --- |
| H1: plain and persona customers give equivalent agent outcomes (surplus share and turns) | Terminus-2 | **Supported.** Plain minus persona: surplus share +0.020 (90% CI -0.020 to +0.061), turns -0.15 (-0.40 to +0.09) |
| | Claude Code | **Inconclusive.** Surplus share is equivalent, +0.006 (-0.027 to +0.043), but turns are not: -0.64 (-0.93 to -0.36), whose interval crosses the -0.8 margin. Episodes with persona customers ran slightly longer |
| H2: agents score higher against an LLM customer than a scripted one | Claude Code | **Supported.** Plain LLM minus scripted surplus share +0.209 (95% CI +0.105 to +0.322) |
| | Terminus-2 | **Supported.** +0.232 (95% CI +0.130 to +0.349) |

Exploratory, not part of the plan:

- The scripted-vs-LLM gap is about twice the one in `equiv-sd3` (+0.08 to
  +0.12 there), where the agents were bare model calls and the customers Opus
  and Sonnet. The harness, the customer model and the seeds all differ, so the
  two runs don't isolate why.
- Persona customers behave differently from plain ones: they open with an
  offer 42% of the time (plain: 100%) and anchor far less hard
  (`cp_max_anchor` 0.35 to 0.40, against about 1.0). The agents' surplus share
  barely moves.

Caveats: one agent model (Sonnet 5.5) and one customer model, both Claude;
the agents answer through MCP tools rather than the JSON reply local runs use,
so these numbers compare with each other, not with `equiv-sd3`'s. Spend: $26.94
for the agents by Harbor's figures (Claude Code $12.78, Terminus-2 about
$14.16) and about $9 for the customers, estimated
from their tokens (the Anthropic API doesn't report cost, so the report's
`Spend` line shows $0.00).

## harbor-user-study (pre-registered)

The analysis plan is the header of
[`configs/harbor-user-study.yaml`](../configs/harbor-user-study.yaml),
committed before any episode existed. Environment: `meeting_scheduling`, an
assistant booking a meeting with a contact; each side has a private calendar
and private preferences. Agent: Harbor's `claude-code` on Claude Sonnet 5.5.
The contact is Claude Haiku 5.5 in both arms, played two ways:

- `aoc`: a direct model call inside the parley-sim sidecar;
- `harbor-user`: Harbor's simulated user (`claude-code` on Haiku 5.5),
  prompting the agent over Harbor's ACP bridge and moving through the
  [counterparty tools](../docs/harbor.md#harbors-simulated-user-as-the-counterpart-experimental).

So the treatment is the whole channel (harness and bridge), not the model.
Seeds 2100 to 2159, one task per seed and arm: 120 trials, all reached the
verifier; 119 completed and verified. Verdict rule: margin, margins 0.08
surplus share and 0.8 turns. Report:
[`harbor-user-study/report.md`](harbor-user-study/report.md).

| Hypothesis | Result |
| --- | --- |
| H1: the two channels give equivalent agent outcomes (surplus share and turns) | **Inconclusive.** Sidecar minus Harbor user, 59 seeds: surplus share +0.062 (90% CI -0.002 to +0.122), whose interval crosses the +0.08 margin; turns +0.39 (+0.02 to +0.78), equivalent |

The agent got 0.800 of the available surplus against the sidecar contact and
0.735 against Harbor's simulated user. Neither "equivalent" nor "different"
(the 95% interval, -0.013 to +0.131, includes zero); the study can't say
whether the channel changes the outcome.

Exploratory, not part of the plan:

- Every completed episode reached a deal in both arms. Joint efficiency
  barely differs (0.931 against 0.926).
- One `harbor-user` trial (seed 2107) did not finish: the simulated user
  started new agent sessions of its own, which had no episode tools, instead
  of prompting the agent, and the episode stalled. It is kept as errored,
  as the plan requires.
- Chat leaks. In this mode everything the agent writes in chat reaches the
  simulated user, outside the episode; the agent is told so on every turn.
  28 of its 237 chat replies state one of its private slot ratings as a
  number, in 21 of the 59 completed trials. Rankings without numbers are
  not counted. The counter was checked by hand on one trial before the run,
  not on this one.
- No `accept` booked a slot other than the one its message named, in either
  arm.
- Harbor's simulated user always opened with its own best slot
  (`cp_first_offer_share`, the contact's value for its first offer as a share
  of its best, 1.000 against 0.989 for the sidecar contact).

Caveats: one agent model and one contact model, both Claude; one environment;
Harbor 0.27. Spend: $4.46 (agent $3.90, simulated user $0.35, sidecar
contact $0.08, two smoke trials outside the run $0.13); the report's `Spend`
line shows $0.00 because the Anthropic API doesn't report cost.

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

For `harbor-pilot`, only the report is published. Its traces include the
persona arm and agent transcripts and are kept privately with the full run.
The same holds for `harbor-user-study`, whose manifest also holds the agent's
chat replies.

`scripts/curate_results.py` made this folder from the full run directories.

## Recomputing

```bash
uv run parley-sim report results/equiv-sd3 --margins focal_surplus_share=0.08,turns=0.8
uv run parley-sim report results/equiv-sd3 results/family-sd4    # joint-report.md
uv run parley-sim verify results/family-sd4/traces/*.jsonl
```

The published reports were produced by the code at the time of each run.
Recomputing with the current code matches them for every comparison that
doesn't involve the withheld arm, with these known differences:

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
  1,048,026 (published: 560 cells, 7,130,265 + 1,239,725);
- `accepted_below_walkaway` now counts only deals the focal seat itself
  accepted. Opus's scripted arm in `equiv-sd3` (and the joint report) moves
  from 0.025 to 0: its one case was Opus proposing a deal below its walk-away,
  which the scripted customer accepted. No verdict changes;
- the report's layout changed: the experimental claim metrics move out of
  the main tables into their own section, the diversity table gains an `n_openings`
  column and each comparison its `n`, the acceptance column reads "decisions
  facing an offer", and p-values below 0.001 print as `p<0.001`;
- an off-by-one in the 90% interval's lower order statistic was fixed. No
  printed `equiv-sd3` figure changes; in the joint report three 90% lower
  bounds move by 0.001 (`cp_first_offer_share` +0.087 to +0.088 and -0.148
  to -0.147, `joint_efficiency` -0.093 to -0.092). No verdict changes.
