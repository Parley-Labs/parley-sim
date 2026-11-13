# family-sd4 (support_desk)

Cells: done=160
Spend: 1,652,556 input + 774,379 output tokens, $7.60 reported by provider

## opus

Means (replicate 0):

| metric | kind | muse | nemotron |
|---|---|---|---|
| deal | relational | +1.000 | +1.000 |
| joint_efficiency | relational | +0.835 | +0.878 |
| focal_surplus_share | relational | +0.551 | +0.659 |
| turns | relational | +7.950 | +6.400 |
| accepted_below_walkaway | intrinsic | +0.000 | +0.000 |
| limit_claim | intrinsic (experimental) | +0.925 | +0.750 |
| understated_limit | intrinsic (experimental) | +0.175 | +0.150 |
| focal_strike_rate | covariate | +0.000 | +0.000 |
| cp_opens_with_offer | counterpart | +0.650 | +0.950 |
| cp_first_offer_share | counterpart | +0.853 | +0.794 |
| cp_max_anchor | counterpart | +0.650 | +0.500 |

_(experimental) metrics come from rules not yet validated against human labels well enough to quote as rates._

nemotron vs muse:

| metric | delta [95% CI] | 90% CI | margin | noise floor | n | verdict |
|---|---|---|---|---|---|---|
| deal | +0.000 [+0.000, +0.000] | [+0.000, +0.000] | - | - | 40 | - |
| joint_efficiency | +0.042 [-0.003, +0.089] | [+0.005, +0.081] | - | - | 40 | - |
| focal_surplus_share | +0.108 [+0.054, +0.166] | [+0.063, +0.158] | - | - | 40 | different |
| turns | -1.550 [-1.900, -1.200] | [-1.850, -1.275] | - | - | 40 | different |
| accepted_below_walkaway | +0.000 [+0.000, +0.000] | [+0.000, +0.000] | - | - | 40 | - |
| limit_claim | -0.175 [-0.350, -0.025] | [-0.300, -0.050] | - | - | 40 | different |
| understated_limit | -0.025 [-0.175, +0.150] | [-0.150, +0.125] | - | - | 40 | - |
| focal_strike_rate | +0.000 [+0.000, +0.000] | [+0.000, +0.000] | - | - | 40 | - |
| cp_opens_with_offer | +0.300 [+0.150, +0.475] | [+0.175, +0.450] | - | - | 40 | different |
| cp_first_offer_share | -0.059 [-0.152, +0.037] | [-0.139, +0.020] | - | - | 40 | - |
| cp_max_anchor | -0.150 [-0.350, +0.050] | [-0.325, +0.025] | - | - | 40 | - |

Diversity (spread across seeds; opening-message variety):

| arm | sd_focal_surplus_share | sd_turns | sd_cp_first_offer_share | opening_distinct2 |
|---|---|---|---|---|
| muse | +0.197 | +0.932 | +0.236 | +0.286 |
| nemotron | +0.197 | +1.081 | +0.229 | +0.229 |

nemotron vs muse: sd_focal_surplus_share -0.000 [-0.051, +0.050]; sd_turns +0.149 [-0.120, +0.410]; sd_cp_first_offer_share -0.007 [-0.059, +0.060]; opening_distinct2 +0.001 (p=0.975)

Acceptance by offer quality (every faced offer, declines included):

| arm | bin | offers | accepted | rate |
|---|---|---|---|---|
| muse | below walkaway | 13 | 0 | 0.00 |
| muse | high | 18 | 9 | 0.50 |
| muse | low | 45 | 2 | 0.04 |
| muse | mid | 57 | 15 | 0.26 |
| nemotron | below walkaway | 7 | 0 | 0.00 |
| nemotron | high | 31 | 14 | 0.45 |
| nemotron | low | 25 | 3 | 0.12 |
| nemotron | mid | 57 | 11 | 0.19 |

## sonnet

Means (replicate 0):

| metric | kind | muse | nemotron |
|---|---|---|---|
| deal | relational | +0.950 | +0.950 |
| joint_efficiency | relational | +0.825 | +0.827 |
| focal_surplus_share | relational | +0.616 | +0.644 |
| turns | relational | +8.300 | +6.775 |
| accepted_below_walkaway | intrinsic | +0.000 | +0.000 |
| limit_claim | intrinsic (experimental) | +0.925 | +0.700 |
| understated_limit | intrinsic (experimental) | +0.400 | +0.225 |
| focal_strike_rate | covariate | +0.000 | +0.000 |
| cp_opens_with_offer | counterpart | +0.800 | +0.975 |
| cp_first_offer_share | counterpart | +0.910 | +0.853 |
| cp_max_anchor | counterpart | +0.825 | +0.575 |

_(experimental) metrics come from rules not yet validated against human labels well enough to quote as rates._

nemotron vs muse:

| metric | delta [95% CI] | 90% CI | margin | noise floor | n | verdict |
|---|---|---|---|---|---|---|
| deal | +0.000 [-0.100, +0.100] | [-0.075, +0.075] | - | - | 40 | - |
| joint_efficiency | +0.002 [-0.089, +0.095] | [-0.075, +0.079] | - | - | 40 | - |
| focal_surplus_share | +0.028 [-0.073, +0.123] | [-0.059, +0.107] | - | - | 40 | - |
| turns | -1.525 [-2.100, -0.925] | [-2.000, -1.025] | - | - | 40 | different |
| accepted_below_walkaway | +0.000 [+0.000, +0.000] | [+0.000, +0.000] | - | - | 40 | - |
| limit_claim | -0.225 [-0.375, -0.075] | [-0.350, -0.100] | - | - | 40 | different |
| understated_limit | -0.175 [-0.325, +0.000] | [-0.300, -0.050] | - | - | 40 | - |
| focal_strike_rate | +0.000 [+0.000, +0.000] | [+0.000, +0.000] | - | - | 40 | - |
| cp_opens_with_offer | +0.175 [+0.050, +0.325] | [+0.075, +0.300] | - | - | 40 | different |
| cp_first_offer_share | -0.057 [-0.150, +0.038] | [-0.136, +0.022] | - | - | 40 | - |
| cp_max_anchor | -0.250 [-0.450, -0.050] | [-0.425, -0.075] | - | - | 40 | different |

Diversity (spread across seeds; opening-message variety):

| arm | sd_focal_surplus_share | sd_turns | sd_cp_first_offer_share | opening_distinct2 |
|---|---|---|---|---|
| muse | +0.243 | +0.992 | +0.201 | +0.254 |
| nemotron | +0.226 | +1.441 | +0.200 | +0.193 |

nemotron vs muse: sd_focal_surplus_share -0.017 [-0.100, +0.069]; sd_turns +0.449 [+0.098, +0.823]; sd_cp_first_offer_share -0.001 [-0.072, +0.086]; opening_distinct2 -0.037 (p=0.261)

Acceptance by offer quality (every faced offer, declines included):

| arm | bin | offers | accepted | rate |
|---|---|---|---|---|
| muse | below walkaway | 21 | 0 | 0.00 |
| muse | high | 23 | 7 | 0.30 |
| muse | low | 31 | 1 | 0.03 |
| muse | mid | 69 | 6 | 0.09 |
| nemotron | below walkaway | 9 | 0 | 0.00 |
| nemotron | high | 25 | 9 | 0.36 |
| nemotron | low | 26 | 0 | 0.00 |
| nemotron | mid | 66 | 13 | 0.20 |

