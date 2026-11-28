# harbor-user-study-harbor (meeting_scheduling)

Cells: done=119, errored=1
Spend: 463,532 input + 76,375 output tokens, $0.00 reported by provider
Verdict rule: margin (replicates averaged per seed; "different" needs a 95% interval excluding zero and, where a margin is set, a delta of at least the margin in either direction)

## claude-code.anthropic-claude-sonnet-5-5

Means (replicate 0; episodes per arm: harbor-user=59, aoc=60; comparisons use only seeds present in both arms, averaging each seed's replicates):

| metric | kind | harbor-user | aoc |
|---|---|---|---|
| deal | relational | +1.000 | +1.000 |
| focal_surplus_share | relational | +0.735 | +0.800 |
| joint_efficiency | relational | +0.926 | +0.931 |
| turns | relational | +6.898 | +7.250 |
| focal_strike_rate | covariate | +0.000 | +0.000 |
| cp_opens_with_offer | counterpart | +1.000 | +1.000 |
| cp_first_offer_share | counterpart | +1.000 | +0.989 |

aoc vs harbor-user:

| metric | delta [95% CI] | 90% CI | margin | noise floor | n | verdict |
|---|---|---|---|---|---|---|
| deal | +0.000 [+0.000, +0.000] | [+0.000, +0.000] | - | - | 59 | - |
| focal_surplus_share | +0.062 [-0.013, +0.131] | [-0.002, +0.122] | +0.080 | - | 59 | inconclusive |
| joint_efficiency | +0.004 [-0.025, +0.033] | [-0.021, +0.029] | - | - | 59 | - |
| turns | +0.390 [-0.068, +0.881] | [+0.017, +0.780] | +0.800 | - | 59 | equivalent |
| focal_strike_rate | +0.000 [+0.000, +0.000] | [+0.000, +0.000] | - | - | 59 | - |
| cp_opens_with_offer | +0.000 [+0.000, +0.000] | [+0.000, +0.000] | - | - | 59 | - |
| cp_first_offer_share | -0.011 [-0.021, -0.004] | [-0.019, -0.004] | - | - | 59 | different |

Diversity (spread across seeds; opening-message variety):

| arm | sd_focal_surplus_share | sd_turns | sd_cp_first_offer_share | opening_distinct2 | n_openings |
|---|---|---|---|---|---|
| harbor-user | +0.261 | +1.269 | +0.000 | +0.187 | 59 |
| aoc | +0.233 | +1.525 | +0.034 | +0.136 | 60 |

aoc vs harbor-user: sd_focal_surplus_share -0.028 [-0.081, +0.026]; sd_turns +0.240 [-0.046, +0.502]; sd_cp_first_offer_share +0.034 [+0.020, +0.044]; opening_distinct2 -0.049 (p=0.004, n=59)
