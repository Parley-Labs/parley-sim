# equiv-sd3 (support_desk)

Cells: done=400
Spend: 5,477,709 input + 465,346 output tokens, $0.00 reported by provider

## opus

Means (replicate 0):

| metric | kind | soc | usersim | aoc |
|---|---|---|---|---|
| deal | relational | +1.000 | +1.000 | +1.000 |
| joint_efficiency | relational | +0.856 | +0.880 | +0.895 |
| focal_surplus_share | relational | +0.566 | +0.671 | +0.679 |
| turns | relational | +7.850 | +6.675 | +6.350 |
| accepted_below_walkaway | intrinsic | +0.025 | +0.000 | +0.000 |
| limit_claim | intrinsic | +0.850 | +0.725 | +0.875 |
| understated_limit | intrinsic | +0.000 | +0.050 | +0.025 |
| focal_strike_rate | covariate | +0.000 | +0.000 | +0.000 |
| cp_opens_with_offer | counterpart | +1.000 | +0.725 | +1.000 |
| cp_first_offer_share | counterpart | +1.000 | +0.851 | +1.000 |
| cp_max_anchor | counterpart | +1.000 | +0.725 | +1.000 |

usersim vs soc:

| metric | delta [95% CI] | 90% CI | margin | noise floor | n | verdict |
|---|---|---|---|---|---|---|
| deal | +0.000 [+0.000, +0.000] | [+0.000, +0.000] | - | +0.000 | 40 | - |
| joint_efficiency | +0.024 [-0.003, +0.053] | [+0.001, +0.047] | - | +0.052 | 40 | - |
| focal_surplus_share | +0.105 [+0.046, +0.161] | [+0.057, +0.152] | +0.080 | +0.106 | 40 | inconclusive |
| turns | -1.175 [-1.650, -0.675] | [-1.575, -0.750] | +0.800 | +0.800 | 40 | different |
| accepted_below_walkaway | -0.025 [-0.075, +0.000] | [-0.075, +0.000] | - | +0.000 | 40 | - |
| limit_claim | -0.125 [-0.250, -0.025] | [-0.225, -0.050] | - | +0.125 | 40 | - |
| understated_limit | +0.050 [+0.000, +0.125] | [+0.000, +0.125] | - | +0.050 | 40 | - |
| focal_strike_rate | +0.000 [+0.000, +0.000] | [+0.000, +0.000] | - | +0.000 | 40 | - |
| cp_opens_with_offer | -0.275 [-0.425, -0.150] | [-0.400, -0.150] | - | +0.000 | 40 | different |
| cp_first_offer_share | -0.149 [-0.229, -0.068] | [-0.216, -0.084] | - | +0.070 | 40 | different |
| cp_max_anchor | -0.275 [-0.425, -0.150] | [-0.400, -0.150] | - | +0.025 | 40 | different |

aoc vs soc:

| metric | delta [95% CI] | 90% CI | margin | noise floor | n | verdict |
|---|---|---|---|---|---|---|
| deal | +0.000 [+0.000, +0.000] | [+0.000, +0.000] | - | +0.000 | 40 | - |
| joint_efficiency | +0.039 [+0.009, +0.071] | [+0.013, +0.066] | - | +0.032 | 40 | different |
| focal_surplus_share | +0.113 [+0.052, +0.173] | [+0.060, +0.163] | +0.080 | +0.088 | 40 | different |
| turns | -1.500 [-1.925, -1.025] | [-1.875, -1.100] | +0.800 | +1.050 | 40 | different |
| accepted_below_walkaway | -0.025 [-0.075, +0.000] | [-0.075, +0.000] | - | +0.000 | 40 | - |
| limit_claim | +0.025 [+0.000, +0.075] | [+0.000, +0.075] | - | +0.100 | 40 | - |
| understated_limit | +0.025 [+0.000, +0.075] | [+0.000, +0.075] | - | +0.075 | 40 | - |
| focal_strike_rate | +0.000 [+0.000, +0.000] | [+0.000, +0.000] | - | +0.000 | 40 | - |
| cp_opens_with_offer | +0.000 [+0.000, +0.000] | [+0.000, +0.000] | - | +0.000 | 40 | - |
| cp_first_offer_share | +0.000 [+0.000, +0.000] | [+0.000, +0.000] | - | +0.000 | 40 | - |
| cp_max_anchor | +0.000 [+0.000, +0.000] | [+0.000, +0.000] | - | +0.000 | 40 | - |

aoc vs usersim:

| metric | delta [95% CI] | 90% CI | margin | noise floor | n | verdict |
|---|---|---|---|---|---|---|
| deal | +0.000 [+0.000, +0.000] | [+0.000, +0.000] | - | +0.000 | 40 | - |
| joint_efficiency | +0.015 [-0.010, +0.042] | [-0.006, +0.038] | - | +0.052 | 40 | - |
| focal_surplus_share | +0.008 [-0.028, +0.047] | [-0.023, +0.041] | +0.080 | +0.106 | 40 | equivalent |
| turns | -0.325 [-0.700, +0.075] | [-0.650, +0.000] | +0.800 | +1.050 | 40 | equivalent |
| accepted_below_walkaway | +0.000 [+0.000, +0.000] | [+0.000, +0.000] | - | +0.000 | 40 | - |
| limit_claim | +0.150 [+0.050, +0.275] | [+0.075, +0.250] | - | +0.125 | 40 | different |
| understated_limit | -0.025 [-0.125, +0.050] | [-0.100, +0.050] | - | +0.075 | 40 | - |
| focal_strike_rate | +0.000 [+0.000, +0.000] | [+0.000, +0.000] | - | +0.000 | 40 | - |
| cp_opens_with_offer | +0.275 [+0.150, +0.425] | [+0.150, +0.400] | - | +0.000 | 40 | different |
| cp_first_offer_share | +0.149 [+0.068, +0.229] | [+0.084, +0.216] | - | +0.070 | 40 | different |
| cp_max_anchor | +0.275 [+0.150, +0.425] | [+0.150, +0.400] | - | +0.025 | 40 | different |

Diversity (spread across seeds; opening-message variety):

| arm | sd_focal_surplus_share | sd_turns | sd_cp_first_offer_share | opening_distinct2 |
|---|---|---|---|---|
| soc | +0.267 | +1.406 | +0.000 | +0.050 |
| usersim | +0.165 | +0.971 | +0.258 | +0.360 |
| aoc | +0.151 | +0.921 | +0.000 | +0.217 |

usersim vs soc: sd_focal_surplus_share -0.103 [-0.138, -0.060]; sd_turns -0.435 [-0.765, -0.052]; sd_cp_first_offer_share +0.258 [+0.177, +0.307]; opening_distinct2 +0.310 (p=0.000)

aoc vs soc: sd_focal_surplus_share -0.116 [-0.155, -0.074]; sd_turns -0.485 [-0.798, -0.122]; sd_cp_first_offer_share +0.000 [+0.000, +0.000]; opening_distinct2 +0.167 (p=0.000)

aoc vs usersim: sd_focal_surplus_share -0.014 [-0.046, +0.020]; sd_turns -0.050 [-0.309, +0.212]; sd_cp_first_offer_share -0.258 [-0.307, -0.177]; opening_distinct2 -0.143 (p=0.000)

Acceptance by offer quality (every faced offer, declines included):

| arm | bin | offers | accepted | rate |
|---|---|---|---|---|
| soc | below walkaway | 35 | 0 | 0.00 |
| soc | high | 11 | 1 | 0.09 |
| soc | low | 56 | 1 | 0.02 |
| soc | mid | 38 | 4 | 0.11 |
| usersim | below walkaway | 8 | 0 | 0.00 |
| usersim | high | 21 | 5 | 0.24 |
| usersim | low | 26 | 0 | 0.00 |
| usersim | mid | 54 | 8 | 0.15 |
| aoc | below walkaway | 14 | 0 | 0.00 |
| aoc | high | 19 | 8 | 0.42 |
| aoc | low | 27 | 0 | 0.00 |
| aoc | mid | 54 | 6 | 0.11 |

## sonnet

Means (replicate 0):

| metric | kind | soc | usersim | aoc |
|---|---|---|---|---|
| deal | relational | +1.000 | +1.000 | +1.000 |
| joint_efficiency | relational | +0.853 | +0.871 | +0.892 |
| focal_surplus_share | relational | +0.611 | +0.691 | +0.713 |
| turns | relational | +7.950 | +6.375 | +6.125 |
| accepted_below_walkaway | intrinsic | +0.000 | +0.000 | +0.000 |
| limit_claim | intrinsic | +0.875 | +0.625 | +0.625 |
| understated_limit | intrinsic | +0.050 | +0.100 | +0.000 |
| focal_strike_rate | covariate | +0.000 | +0.000 | +0.000 |
| cp_opens_with_offer | counterpart | +1.000 | +0.725 | +1.000 |
| cp_first_offer_share | counterpart | +1.000 | +0.828 | +1.000 |
| cp_max_anchor | counterpart | +1.000 | +0.700 | +1.000 |

usersim vs soc:

| metric | delta [95% CI] | 90% CI | margin | noise floor | n | verdict |
|---|---|---|---|---|---|---|
| deal | +0.000 [+0.000, +0.000] | [+0.000, +0.000] | - | +0.000 | 40 | - |
| joint_efficiency | +0.017 [-0.007, +0.043] | [-0.003, +0.038] | - | +0.041 | 40 | - |
| focal_surplus_share | +0.080 [+0.026, +0.135] | [+0.035, +0.125] | +0.080 | +0.082 | 40 | inconclusive |
| turns | -1.575 [-2.075, -1.075] | [-1.975, -1.150] | +0.800 | +0.700 | 40 | different |
| accepted_below_walkaway | +0.000 [+0.000, +0.000] | [+0.000, +0.000] | - | +0.000 | 40 | - |
| limit_claim | -0.250 [-0.425, -0.100] | [-0.400, -0.125] | - | +0.250 | 40 | - |
| understated_limit | +0.050 [-0.075, +0.150] | [-0.050, +0.150] | - | +0.100 | 40 | - |
| focal_strike_rate | +0.000 [+0.000, +0.000] | [+0.000, +0.000] | - | +0.000 | 40 | - |
| cp_opens_with_offer | -0.275 [-0.425, -0.150] | [-0.400, -0.150] | - | +0.025 | 40 | different |
| cp_first_offer_share | -0.172 [-0.268, -0.086] | [-0.250, -0.099] | - | +0.089 | 40 | different |
| cp_max_anchor | -0.300 [-0.450, -0.150] | [-0.425, -0.175] | - | +0.100 | 40 | different |

aoc vs soc:

| metric | delta [95% CI] | 90% CI | margin | noise floor | n | verdict |
|---|---|---|---|---|---|---|
| deal | +0.000 [+0.000, +0.000] | [+0.000, +0.000] | - | +0.000 | 40 | - |
| joint_efficiency | +0.038 [+0.013, +0.067] | [+0.016, +0.062] | - | +0.042 | 40 | - |
| focal_surplus_share | +0.102 [+0.051, +0.151] | [+0.058, +0.144] | +0.080 | +0.093 | 40 | different |
| turns | -1.825 [-2.425, -1.200] | [-2.325, -1.300] | +0.800 | +0.875 | 40 | different |
| accepted_below_walkaway | +0.000 [+0.000, +0.000] | [+0.000, +0.000] | - | +0.000 | 40 | - |
| limit_claim | -0.250 [-0.400, -0.100] | [-0.375, -0.125] | - | +0.325 | 40 | - |
| understated_limit | -0.050 [-0.125, +0.000] | [-0.100, +0.000] | - | +0.000 | 40 | - |
| focal_strike_rate | +0.000 [+0.000, +0.000] | [+0.000, +0.000] | - | +0.000 | 40 | - |
| cp_opens_with_offer | +0.000 [+0.000, +0.000] | [+0.000, +0.000] | - | +0.000 | 40 | - |
| cp_first_offer_share | +0.000 [+0.000, +0.000] | [+0.000, +0.000] | - | +0.000 | 40 | - |
| cp_max_anchor | +0.000 [+0.000, +0.000] | [+0.000, +0.000] | - | +0.000 | 40 | - |

aoc vs usersim:

| metric | delta [95% CI] | 90% CI | margin | noise floor | n | verdict |
|---|---|---|---|---|---|---|
| deal | +0.000 [+0.000, +0.000] | [+0.000, +0.000] | - | +0.000 | 40 | - |
| joint_efficiency | +0.021 [-0.002, +0.045] | [+0.002, +0.042] | - | +0.042 | 40 | - |
| focal_surplus_share | +0.022 [-0.030, +0.073] | [-0.022, +0.065] | +0.080 | +0.093 | 40 | equivalent |
| turns | -0.250 [-0.675, +0.150] | [-0.625, +0.100] | +0.800 | +0.875 | 40 | equivalent |
| accepted_below_walkaway | +0.000 [+0.000, +0.000] | [+0.000, +0.000] | - | +0.000 | 40 | - |
| limit_claim | +0.000 [-0.150, +0.150] | [-0.125, +0.125] | - | +0.325 | 40 | - |
| understated_limit | -0.100 [-0.200, -0.025] | [-0.175, -0.025] | - | +0.100 | 40 | - |
| focal_strike_rate | +0.000 [+0.000, +0.000] | [+0.000, +0.000] | - | +0.000 | 40 | - |
| cp_opens_with_offer | +0.275 [+0.150, +0.425] | [+0.150, +0.400] | - | +0.025 | 40 | different |
| cp_first_offer_share | +0.172 [+0.086, +0.268] | [+0.099, +0.249] | - | +0.089 | 40 | different |
| cp_max_anchor | +0.300 [+0.150, +0.450] | [+0.175, +0.425] | - | +0.100 | 40 | different |

Diversity (spread across seeds; opening-message variety):

| arm | sd_focal_surplus_share | sd_turns | sd_cp_first_offer_share | opening_distinct2 |
|---|---|---|---|---|
| soc | +0.232 | +1.484 | +0.000 | +0.050 |
| usersim | +0.179 | +1.030 | +0.293 | +0.358 |
| aoc | +0.164 | +1.159 | +0.000 | +0.217 |

usersim vs soc: sd_focal_surplus_share -0.052 [-0.096, -0.008]; sd_turns -0.454 [-0.761, -0.059]; sd_cp_first_offer_share +0.293 [+0.201, +0.372]; opening_distinct2 +0.308 (p=0.000)

aoc vs soc: sd_focal_surplus_share -0.068 [-0.099, -0.035]; sd_turns -0.325 [-0.717, +0.092]; sd_cp_first_offer_share +0.000 [+0.000, +0.000]; opening_distinct2 +0.167 (p=0.000)

aoc vs usersim: sd_focal_surplus_share -0.015 [-0.060, +0.028]; sd_turns +0.129 [-0.156, +0.375]; sd_cp_first_offer_share -0.293 [-0.372, -0.201]; opening_distinct2 -0.140 (p=0.000)

Acceptance by offer quality (every faced offer, declines included):

| arm | bin | offers | accepted | rate |
|---|---|---|---|---|
| soc | below walkaway | 36 | 0 | 0.00 |
| soc | high | 13 | 3 | 0.23 |
| soc | low | 56 | 0 | 0.00 |
| soc | mid | 37 | 3 | 0.08 |
| usersim | below walkaway | 9 | 0 | 0.00 |
| usersim | high | 21 | 5 | 0.24 |
| usersim | low | 27 | 0 | 0.00 |
| usersim | mid | 46 | 8 | 0.17 |
| aoc | below walkaway | 14 | 0 | 0.00 |
| aoc | high | 23 | 10 | 0.43 |
| aoc | low | 33 | 0 | 0.00 |
| aoc | mid | 40 | 5 | 0.12 |

