# Concepts

## Seats, focal seat and counterparts

An environment has seats (in `support_desk`: `agent` and `customer`). One seat
is the **focal** seat: the agent under test. Everything else is a
**counterpart**. A seat is filled by one of:

- `policy:<name>`: a scripted policy that runs in-process (`hardliner`,
  `reasonable`, `conceder` in `support_desk`);
- `model:<name>`: a model from the config's `lineup`, called through its
  provider.

## Arms

An arm is one way of filling the counterpart seats. Every run has two, and
can add more:

| Arm | Counterparts | Config field |
| --- | --- | --- |
| `soc` | Scripted counterparts (scripted-opponent condition) | `scripted_counterparts` |
| `aoc` | Model counterparts (adaptive-opponent condition) | `adaptive_counterparts` |
| any name | Any other model counterparts, e.g. persona-driven users or another model family | `extra_arms` |

When an arm lists several counterparts, they rotate across seeds, so each is
covered even when the environment has a single counterpart seat.

## Matched pairs

For every seed and focal model, parley-sim builds one cell per arm. The cells
share the seed, the generated scenario and the focal seat; only the
counterpart differs. They share a `pair_key`, and the report compares arms
**within** pairs, never pooled means. A difference between arms is therefore
a difference caused by the counterpart.

Arms can run in separate directories, for example when you add an arm to an
earlier run. `parley-sim report DIR1 DIR2` joins them on seed, focal model,
arm and replicate, and refuses to join runs whose scenarios differ.

## The noise floor

Model counterparts are not deterministic. To know how much results move by
chance alone, every model-counterpart cell is played again with an identical
setup (`self_null_replicates`, default 1). The **noise floor** of a metric is
the mean absolute difference between those replays. For a comparison between
two arms, the larger of the two arms' floors applies.

## Verdicts

Each comparison reports, per metric:

| Column | Meaning |
| --- | --- |
| delta [95% CI] | Mean paired difference (second arm minus first, as titled), with a bootstrap 95% interval |
| 90% CI | The same at 90%, used for equivalence |
| margin | The equivalence margin you passed with `--margins`, if any |
| noise floor | See above |
| n | Number of matched pairs |
| verdict | See below |

The verdict rules:

- **equivalent**: a margin was given and the 90% interval lies entirely
  within plus or minus the margin (two one-sided tests at 5%);
- **different**: the 95% interval excludes zero **and** the mean difference
  is larger than the noise floor;
- **inconclusive**: a margin was given and neither of the above holds;
- **-**: no margin was given and the difference is not established.

A difference can exclude zero and still not count, if it is no larger than
what replays of the same setup produce. Set margins before you look at the
data: the bundled `configs/equiv-sd3.yaml` shows a pre-registered plan.

## Metric kinds

Every metric declares what it is about, so a reader knows what a difference
means:

| Kind | Meaning | Examples in `support_desk` |
| --- | --- | --- |
| intrinsic | The focal seat's own behavior, given what it faced | `accepted_below_walkaway` |
| relational | Only exists through the interaction | `deal`, `focal_surplus_share`, `joint_efficiency`, `turns` |
| covariate | Recorded so differences can be attributed | `focal_strike_rate` |
| counterpart | How the other side behaved | `cp_opens_with_offer`, `cp_first_offer_share`, `cp_max_anchor` |

A kind ending in `(experimental)` comes from rules that are not yet validated
well enough to quote as a rate (see [claim metrics](claims.md)).

Counterpart metrics matter more than they look. If two arms differ on the
agent's outcome, the counterpart metrics show whether the counterparts
behaved differently, which is usually the explanation.

## Diversity

The report also measures how varied each arm is:

- `sd_<metric>`: the standard deviation of a metric across seeds, compared
  between arms with a paired bootstrap interval;
- `opening_distinct2`: the share of distinct word pairs in the counterpart's
  opening messages across seeds, compared with a paired permutation test
  (p-value).

A uniform counterpart (every scripted bot, and some LLMs) gives zero spread on
its own behavior even when the agent's results vary.

## Acceptance by offer quality

Logging only accepted deals hides what an agent turned down. The report bins
every offer the focal seat faced by its quality (its value to the focal seat
relative to the best it could get, below walk-away to high) and shows the
acceptance rate per bin and arm. Two arms can differ in acceptance simply
because one kind of counterpart makes better offers; this table separates
that from a change in the agent.

## Strikes and fallbacks

When a model's output does not parse, or proposes an illegal move, the
episode records a **strike** with the reason and applies the environment's
fallback move. Strikes never end an episode on their own. A high
`focal_strike_rate` in one arm is a reason to distrust that arm's comparison.

## Traces

Each episode writes one JSONL file: a `header` (spec, scenario, ground
truth), then `decision` and `event` lines in order, then a `footer`
(outcomes, final state). Every decision carries its observation, its context
(what it was answering), what was submitted and what was applied, plus any
strike and the provider's raw response metadata. A footer is written even
when an episode errors, so a missing footer means a truncated file.
`parley-sim verify` replays a trace through the environment and checks that
every state hash matches.
