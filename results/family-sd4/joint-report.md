# equiv-sd3 + family-sd4 (support_desk)

Cells: done=560
Spend: 7,130,265 input + 1,239,725 output tokens, $7.60 reported by provider

## opus

Means (replicate 0):

| metric | kind | soc | usersim | muse | nemotron | aoc |
|---|---|---|---|---|---|---|
| deal | relational | +1.000 | +1.000 | +1.000 | +1.000 | +1.000 |
| joint_efficiency | relational | +0.856 | +0.880 | +0.835 | +0.878 | +0.895 |
| focal_surplus_share | relational | +0.566 | +0.671 | +0.551 | +0.659 | +0.679 |
| turns | relational | +7.850 | +6.675 | +7.950 | +6.400 | +6.350 |
| accepted_below_walkaway | intrinsic | +0.025 | +0.000 | +0.000 | +0.000 | +0.000 |
| limit_claim | intrinsic (experimental) | +0.900 | +0.775 | +0.925 | +0.750 | +0.875 |
| understated_limit | intrinsic (experimental) | +0.150 | +0.125 | +0.175 | +0.150 | +0.175 |
| focal_strike_rate | covariate | +0.000 | +0.000 | +0.000 | +0.000 | +0.000 |
| cp_opens_with_offer | counterpart | +1.000 | +0.725 | +0.650 | +0.950 | +1.000 |
| cp_first_offer_share | counterpart | +1.000 | +0.851 | +0.853 | +0.794 | +1.000 |
| cp_max_anchor | counterpart | +1.000 | +0.725 | +0.650 | +0.500 | +1.000 |

_(experimental) metrics come from rules not yet validated against human labels well enough to quote as rates._

usersim vs soc:

| metric | delta [95% CI] | 90% CI | margin | noise floor | n | verdict |
|---|---|---|---|---|---|---|
| deal | +0.000 [+0.000, +0.000] | [+0.000, +0.000] | - | +0.000 | 40 | - |
| joint_efficiency | +0.024 [-0.003, +0.053] | [+0.001, +0.047] | - | +0.052 | 40 | - |
| focal_surplus_share | +0.105 [+0.046, +0.161] | [+0.057, +0.152] | - | +0.106 | 40 | - |
| turns | -1.175 [-1.650, -0.675] | [-1.575, -0.750] | - | +0.800 | 40 | different |
| accepted_below_walkaway | -0.025 [-0.075, +0.000] | [-0.075, +0.000] | - | +0.000 | 40 | - |
| limit_claim | -0.125 [-0.225, -0.025] | [-0.225, -0.050] | - | +0.150 | 40 | - |
| understated_limit | -0.025 [-0.175, +0.125] | [-0.150, +0.100] | - | +0.275 | 40 | - |
| focal_strike_rate | +0.000 [+0.000, +0.000] | [+0.000, +0.000] | - | +0.000 | 40 | - |
| cp_opens_with_offer | -0.275 [-0.425, -0.150] | [-0.400, -0.150] | - | +0.000 | 40 | different |
| cp_first_offer_share | -0.149 [-0.229, -0.068] | [-0.216, -0.084] | - | +0.070 | 40 | different |
| cp_max_anchor | -0.275 [-0.425, -0.150] | [-0.400, -0.150] | - | +0.025 | 40 | different |

muse vs soc:

| metric | delta [95% CI] | 90% CI | margin | noise floor | n | verdict |
|---|---|---|---|---|---|---|
| deal | +0.000 [+0.000, +0.000] | [+0.000, +0.000] | - | - | 40 | - |
| joint_efficiency | -0.020 [-0.058, +0.011] | [-0.051, +0.006] | - | - | 40 | - |
| focal_surplus_share | -0.015 [-0.066, +0.035] | [-0.057, +0.025] | - | - | 40 | - |
| turns | +0.100 [-0.375, +0.650] | [-0.300, +0.550] | - | - | 40 | - |
| accepted_below_walkaway | -0.025 [-0.075, +0.000] | [-0.075, +0.000] | - | - | 40 | - |
| limit_claim | +0.025 [-0.075, +0.125] | [-0.075, +0.125] | - | - | 40 | - |
| understated_limit | +0.025 [-0.125, +0.175] | [-0.100, +0.150] | - | - | 40 | - |
| focal_strike_rate | +0.000 [+0.000, +0.000] | [+0.000, +0.000] | - | - | 40 | - |
| cp_opens_with_offer | -0.350 [-0.500, -0.200] | [-0.475, -0.225] | - | - | 40 | different |
| cp_first_offer_share | -0.147 [-0.223, -0.078] | [-0.210, -0.088] | - | - | 40 | different |
| cp_max_anchor | -0.350 [-0.500, -0.200] | [-0.475, -0.225] | - | - | 40 | different |

nemotron vs soc:

| metric | delta [95% CI] | 90% CI | margin | noise floor | n | verdict |
|---|---|---|---|---|---|---|
| deal | +0.000 [+0.000, +0.000] | [+0.000, +0.000] | - | - | 40 | - |
| joint_efficiency | +0.022 [-0.012, +0.056] | [-0.007, +0.051] | - | - | 40 | - |
| focal_surplus_share | +0.093 [+0.027, +0.163] | [+0.037, +0.154] | - | - | 40 | different |
| turns | -1.450 [-1.950, -0.900] | [-1.875, -1.000] | - | - | 40 | different |
| accepted_below_walkaway | -0.025 [-0.075, +0.000] | [-0.075, +0.000] | - | - | 40 | - |
| limit_claim | -0.150 [-0.275, -0.025] | [-0.275, -0.050] | - | - | 40 | different |
| understated_limit | +0.000 [-0.175, +0.175] | [-0.150, +0.150] | - | - | 40 | - |
| focal_strike_rate | +0.000 [+0.000, +0.000] | [+0.000, +0.000] | - | - | 40 | - |
| cp_opens_with_offer | -0.050 [-0.125, +0.000] | [-0.100, +0.000] | - | - | 40 | - |
| cp_first_offer_share | -0.206 [-0.275, -0.139] | [-0.264, -0.148] | - | - | 40 | different |
| cp_max_anchor | -0.500 [-0.650, -0.350] | [-0.625, -0.375] | - | - | 40 | different |

aoc vs soc:

| metric | delta [95% CI] | 90% CI | margin | noise floor | n | verdict |
|---|---|---|---|---|---|---|
| deal | +0.000 [+0.000, +0.000] | [+0.000, +0.000] | - | +0.000 | 40 | - |
| joint_efficiency | +0.039 [+0.009, +0.071] | [+0.013, +0.066] | - | +0.032 | 40 | different |
| focal_surplus_share | +0.113 [+0.052, +0.173] | [+0.060, +0.163] | - | +0.088 | 40 | different |
| turns | -1.500 [-1.925, -1.025] | [-1.875, -1.100] | - | +1.050 | 40 | different |
| accepted_below_walkaway | -0.025 [-0.075, +0.000] | [-0.075, +0.000] | - | +0.000 | 40 | - |
| limit_claim | -0.025 [-0.100, +0.050] | [-0.100, +0.050] | - | +0.100 | 40 | - |
| understated_limit | +0.025 [-0.150, +0.200] | [-0.125, +0.175] | - | +0.325 | 40 | - |
| focal_strike_rate | +0.000 [+0.000, +0.000] | [+0.000, +0.000] | - | +0.000 | 40 | - |
| cp_opens_with_offer | +0.000 [+0.000, +0.000] | [+0.000, +0.000] | - | +0.000 | 40 | - |
| cp_first_offer_share | +0.000 [+0.000, +0.000] | [+0.000, +0.000] | - | +0.000 | 40 | - |
| cp_max_anchor | +0.000 [+0.000, +0.000] | [+0.000, +0.000] | - | +0.000 | 40 | - |

muse vs usersim:

| metric | delta [95% CI] | 90% CI | margin | noise floor | n | verdict |
|---|---|---|---|---|---|---|
| deal | +0.000 [+0.000, +0.000] | [+0.000, +0.000] | - | +0.000 | 40 | - |
| joint_efficiency | -0.044 [-0.088, -0.004] | [-0.082, -0.011] | - | +0.052 | 40 | - |
| focal_surplus_share | -0.120 [-0.167, -0.077] | [-0.159, -0.085] | - | +0.106 | 40 | different |
| turns | +1.275 [+0.875, +1.650] | [+0.950, +1.600] | - | +0.800 | 40 | different |
| accepted_below_walkaway | +0.000 [+0.000, +0.000] | [+0.000, +0.000] | - | +0.000 | 40 | - |
| limit_claim | +0.150 [+0.025, +0.275] | [+0.050, +0.250] | - | +0.150 | 40 | - |
| understated_limit | +0.050 [-0.125, +0.200] | [-0.075, +0.175] | - | +0.275 | 40 | - |
| focal_strike_rate | +0.000 [+0.000, +0.000] | [+0.000, +0.000] | - | +0.000 | 40 | - |
| cp_opens_with_offer | -0.075 [-0.300, +0.125] | [-0.250, +0.100] | - | +0.000 | 40 | - |
| cp_first_offer_share | +0.002 [-0.109, +0.113] | [-0.092, +0.096] | - | +0.070 | 40 | - |
| cp_max_anchor | -0.075 [-0.275, +0.125] | [-0.250, +0.100] | - | +0.025 | 40 | - |

nemotron vs usersim:

| metric | delta [95% CI] | 90% CI | margin | noise floor | n | verdict |
|---|---|---|---|---|---|---|
| deal | +0.000 [+0.000, +0.000] | [+0.000, +0.000] | - | +0.000 | 40 | - |
| joint_efficiency | -0.002 [-0.030, +0.023] | [-0.026, +0.019] | - | +0.052 | 40 | - |
| focal_surplus_share | -0.012 [-0.056, +0.037] | [-0.051, +0.029] | - | +0.106 | 40 | - |
| turns | -0.275 [-0.625, +0.075] | [-0.575, +0.025] | - | +0.800 | 40 | - |
| accepted_below_walkaway | +0.000 [+0.000, +0.000] | [+0.000, +0.000] | - | +0.000 | 40 | - |
| limit_claim | -0.025 [-0.175, +0.125] | [-0.150, +0.100] | - | +0.150 | 40 | - |
| understated_limit | +0.025 [-0.125, +0.175] | [-0.100, +0.150] | - | +0.275 | 40 | - |
| focal_strike_rate | +0.000 [+0.000, +0.000] | [+0.000, +0.000] | - | +0.000 | 40 | - |
| cp_opens_with_offer | +0.225 [+0.075, +0.375] | [+0.100, +0.350] | - | +0.000 | 40 | different |
| cp_first_offer_share | -0.057 [-0.167, +0.051] | [-0.150, +0.035] | - | +0.070 | 40 | - |
| cp_max_anchor | -0.225 [-0.425, -0.025] | [-0.400, -0.050] | - | +0.025 | 40 | different |

aoc vs usersim:

| metric | delta [95% CI] | 90% CI | margin | noise floor | n | verdict |
|---|---|---|---|---|---|---|
| deal | +0.000 [+0.000, +0.000] | [+0.000, +0.000] | - | +0.000 | 40 | - |
| joint_efficiency | +0.015 [-0.010, +0.042] | [-0.006, +0.038] | - | +0.052 | 40 | - |
| focal_surplus_share | +0.008 [-0.028, +0.047] | [-0.023, +0.041] | - | +0.106 | 40 | - |
| turns | -0.325 [-0.700, +0.075] | [-0.650, +0.000] | - | +1.050 | 40 | - |
| accepted_below_walkaway | +0.000 [+0.000, +0.000] | [+0.000, +0.000] | - | +0.000 | 40 | - |
| limit_claim | +0.100 [+0.025, +0.200] | [+0.025, +0.175] | - | +0.150 | 40 | - |
| understated_limit | +0.050 [-0.100, +0.200] | [-0.075, +0.175] | - | +0.325 | 40 | - |
| focal_strike_rate | +0.000 [+0.000, +0.000] | [+0.000, +0.000] | - | +0.000 | 40 | - |
| cp_opens_with_offer | +0.275 [+0.150, +0.425] | [+0.150, +0.400] | - | +0.000 | 40 | different |
| cp_first_offer_share | +0.149 [+0.068, +0.229] | [+0.084, +0.216] | - | +0.070 | 40 | different |
| cp_max_anchor | +0.275 [+0.150, +0.425] | [+0.150, +0.400] | - | +0.025 | 40 | different |

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

aoc vs muse:

| metric | delta [95% CI] | 90% CI | margin | noise floor | n | verdict |
|---|---|---|---|---|---|---|
| deal | +0.000 [+0.000, +0.000] | [+0.000, +0.000] | - | +0.000 | 40 | - |
| joint_efficiency | +0.059 [+0.019, +0.103] | [+0.024, +0.096] | - | +0.032 | 40 | different |
| focal_surplus_share | +0.128 [+0.082, +0.178] | [+0.088, +0.169] | - | +0.088 | 40 | different |
| turns | -1.600 [-1.950, -1.250] | [-1.900, -1.300] | - | +1.050 | 40 | different |
| accepted_below_walkaway | +0.000 [+0.000, +0.000] | [+0.000, +0.000] | - | +0.000 | 40 | - |
| limit_claim | -0.050 [-0.150, +0.025] | [-0.125, +0.025] | - | +0.100 | 40 | - |
| understated_limit | +0.000 [-0.150, +0.175] | [-0.150, +0.150] | - | +0.325 | 40 | - |
| focal_strike_rate | +0.000 [+0.000, +0.000] | [+0.000, +0.000] | - | +0.000 | 40 | - |
| cp_opens_with_offer | +0.350 [+0.200, +0.500] | [+0.225, +0.475] | - | +0.000 | 40 | different |
| cp_first_offer_share | +0.147 [+0.078, +0.223] | [+0.087, +0.210] | - | +0.000 | 40 | different |
| cp_max_anchor | +0.350 [+0.200, +0.500] | [+0.225, +0.475] | - | +0.000 | 40 | different |

aoc vs nemotron:

| metric | delta [95% CI] | 90% CI | margin | noise floor | n | verdict |
|---|---|---|---|---|---|---|
| deal | +0.000 [+0.000, +0.000] | [+0.000, +0.000] | - | +0.000 | 40 | - |
| joint_efficiency | +0.017 [-0.009, +0.045] | [-0.005, +0.042] | - | +0.032 | 40 | - |
| focal_surplus_share | +0.020 [-0.034, +0.074] | [-0.025, +0.065] | - | +0.088 | 40 | - |
| turns | -0.050 [-0.475, +0.375] | [-0.400, +0.275] | - | +1.050 | 40 | - |
| accepted_below_walkaway | +0.000 [+0.000, +0.000] | [+0.000, +0.000] | - | +0.000 | 40 | - |
| limit_claim | +0.125 [+0.000, +0.250] | [+0.025, +0.250] | - | +0.100 | 40 | - |
| understated_limit | +0.025 [-0.125, +0.175] | [-0.100, +0.150] | - | +0.325 | 40 | - |
| focal_strike_rate | +0.000 [+0.000, +0.000] | [+0.000, +0.000] | - | +0.000 | 40 | - |
| cp_opens_with_offer | +0.050 [+0.000, +0.125] | [+0.000, +0.100] | - | +0.000 | 40 | - |
| cp_first_offer_share | +0.206 [+0.139, +0.275] | [+0.148, +0.264] | - | +0.000 | 40 | different |
| cp_max_anchor | +0.500 [+0.350, +0.650] | [+0.375, +0.625] | - | +0.000 | 40 | different |

Diversity (spread across seeds; opening-message variety):

| arm | sd_focal_surplus_share | sd_turns | sd_cp_first_offer_share | opening_distinct2 |
|---|---|---|---|---|
| soc | +0.267 | +1.406 | +0.000 | +0.050 |
| usersim | +0.165 | +0.971 | +0.258 | +0.360 |
| muse | +0.197 | +0.932 | +0.236 | +0.286 |
| nemotron | +0.197 | +1.081 | +0.229 | +0.229 |
| aoc | +0.151 | +0.921 | +0.000 | +0.217 |

usersim vs soc: sd_focal_surplus_share -0.103 [-0.138, -0.060]; sd_turns -0.435 [-0.765, -0.052]; sd_cp_first_offer_share +0.258 [+0.177, +0.307]; opening_distinct2 +0.310 (p=0.000)

muse vs soc: sd_focal_surplus_share -0.070 [-0.106, -0.031]; sd_turns -0.474 [-0.822, -0.100]; sd_cp_first_offer_share +0.236 [+0.170, +0.279]; opening_distinct2 +0.209 (p=0.000)

nemotron vs soc: sd_focal_surplus_share -0.070 [-0.117, -0.018]; sd_turns -0.325 [-0.643, +0.044]; sd_cp_first_offer_share +0.229 [+0.197, +0.250]; opening_distinct2 +0.179 (p=0.000)

aoc vs soc: sd_focal_surplus_share -0.116 [-0.155, -0.074]; sd_turns -0.485 [-0.798, -0.122]; sd_cp_first_offer_share +0.000 [+0.000, +0.000]; opening_distinct2 +0.167 (p=0.000)

muse vs usersim: sd_focal_surplus_share +0.033 [-0.006, +0.069]; sd_turns -0.039 [-0.344, +0.247]; sd_cp_first_offer_share -0.022 [-0.110, +0.064]; opening_distinct2 -0.129 (p=0.001)

nemotron vs usersim: sd_focal_surplus_share +0.033 [-0.000, +0.065]; sd_turns +0.110 [-0.155, +0.352]; sd_cp_first_offer_share -0.030 [-0.093, +0.055]; opening_distinct2 -0.131 (p=0.000)

aoc vs usersim: sd_focal_surplus_share -0.014 [-0.046, +0.020]; sd_turns -0.050 [-0.309, +0.212]; sd_cp_first_offer_share -0.258 [-0.307, -0.177]; opening_distinct2 -0.143 (p=0.000)

nemotron vs muse: sd_focal_surplus_share -0.000 [-0.051, +0.050]; sd_turns +0.149 [-0.120, +0.410]; sd_cp_first_offer_share -0.007 [-0.059, +0.060]; opening_distinct2 +0.001 (p=0.975)

aoc vs muse: sd_focal_surplus_share -0.047 [-0.087, -0.001]; sd_turns -0.011 [-0.285, +0.285]; sd_cp_first_offer_share -0.236 [-0.279, -0.170]; opening_distinct2 -0.008 (p=0.780)

aoc vs nemotron: sd_focal_surplus_share -0.046 [-0.086, -0.007]; sd_turns -0.160 [-0.394, +0.064]; sd_cp_first_offer_share -0.229 [-0.250, -0.197]; opening_distinct2 -0.012 (p=0.682)

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
| muse | below walkaway | 13 | 0 | 0.00 |
| muse | high | 18 | 9 | 0.50 |
| muse | low | 45 | 2 | 0.04 |
| muse | mid | 57 | 15 | 0.26 |
| nemotron | below walkaway | 7 | 0 | 0.00 |
| nemotron | high | 31 | 14 | 0.45 |
| nemotron | low | 25 | 3 | 0.12 |
| nemotron | mid | 57 | 11 | 0.19 |
| aoc | below walkaway | 14 | 0 | 0.00 |
| aoc | high | 19 | 8 | 0.42 |
| aoc | low | 27 | 0 | 0.00 |
| aoc | mid | 54 | 6 | 0.11 |

## sonnet

Means (replicate 0):

| metric | kind | soc | usersim | muse | nemotron | aoc |
|---|---|---|---|---|---|---|
| deal | relational | +1.000 | +1.000 | +0.950 | +0.950 | +1.000 |
| joint_efficiency | relational | +0.853 | +0.871 | +0.825 | +0.827 | +0.892 |
| focal_surplus_share | relational | +0.611 | +0.691 | +0.616 | +0.644 | +0.713 |
| turns | relational | +7.950 | +6.375 | +8.300 | +6.775 | +6.125 |
| accepted_below_walkaway | intrinsic | +0.000 | +0.000 | +0.000 | +0.000 | +0.000 |
| limit_claim | intrinsic (experimental) | +0.975 | +0.750 | +0.925 | +0.700 | +0.700 |
| understated_limit | intrinsic (experimental) | +0.275 | +0.300 | +0.400 | +0.225 | +0.350 |
| focal_strike_rate | covariate | +0.000 | +0.000 | +0.000 | +0.000 | +0.000 |
| cp_opens_with_offer | counterpart | +1.000 | +0.725 | +0.800 | +0.975 | +1.000 |
| cp_first_offer_share | counterpart | +1.000 | +0.828 | +0.910 | +0.853 | +1.000 |
| cp_max_anchor | counterpart | +1.000 | +0.700 | +0.825 | +0.575 | +1.000 |

_(experimental) metrics come from rules not yet validated against human labels well enough to quote as rates._

usersim vs soc:

| metric | delta [95% CI] | 90% CI | margin | noise floor | n | verdict |
|---|---|---|---|---|---|---|
| deal | +0.000 [+0.000, +0.000] | [+0.000, +0.000] | - | +0.000 | 40 | - |
| joint_efficiency | +0.017 [-0.007, +0.043] | [-0.003, +0.038] | - | +0.041 | 40 | - |
| focal_surplus_share | +0.080 [+0.026, +0.135] | [+0.035, +0.125] | - | +0.082 | 40 | - |
| turns | -1.575 [-2.075, -1.075] | [-1.975, -1.150] | - | +0.700 | 40 | different |
| accepted_below_walkaway | +0.000 [+0.000, +0.000] | [+0.000, +0.000] | - | +0.000 | 40 | - |
| limit_claim | -0.225 [-0.375, -0.075] | [-0.350, -0.100] | - | +0.075 | 40 | different |
| understated_limit | +0.025 [-0.175, +0.200] | [-0.125, +0.175] | - | +0.350 | 40 | - |
| focal_strike_rate | +0.000 [+0.000, +0.000] | [+0.000, +0.000] | - | +0.000 | 40 | - |
| cp_opens_with_offer | -0.275 [-0.425, -0.150] | [-0.400, -0.150] | - | +0.025 | 40 | different |
| cp_first_offer_share | -0.172 [-0.268, -0.086] | [-0.250, -0.099] | - | +0.089 | 40 | different |
| cp_max_anchor | -0.300 [-0.450, -0.150] | [-0.425, -0.175] | - | +0.100 | 40 | different |

muse vs soc:

| metric | delta [95% CI] | 90% CI | margin | noise floor | n | verdict |
|---|---|---|---|---|---|---|
| deal | -0.050 [-0.125, +0.000] | [-0.100, +0.000] | - | - | 40 | - |
| joint_efficiency | -0.029 [-0.102, +0.029] | [-0.090, +0.023] | - | - | 40 | - |
| focal_surplus_share | +0.004 [-0.082, +0.091] | [-0.070, +0.079] | - | - | 40 | - |
| turns | +0.350 [-0.175, +0.875] | [-0.100, +0.800] | - | - | 40 | - |
| accepted_below_walkaway | +0.000 [+0.000, +0.000] | [+0.000, +0.000] | - | - | 40 | - |
| limit_claim | -0.050 [-0.150, +0.050] | [-0.150, +0.025] | - | - | 40 | - |
| understated_limit | +0.125 [-0.050, +0.300] | [-0.025, +0.275] | - | - | 40 | - |
| focal_strike_rate | +0.000 [+0.000, +0.000] | [+0.000, +0.000] | - | - | 40 | - |
| cp_opens_with_offer | -0.200 [-0.325, -0.075] | [-0.300, -0.100] | - | - | 40 | different |
| cp_first_offer_share | -0.090 [-0.157, -0.033] | [-0.148, -0.041] | - | - | 40 | different |
| cp_max_anchor | -0.175 [-0.300, -0.075] | [-0.275, -0.075] | - | - | 40 | different |

nemotron vs soc:

| metric | delta [95% CI] | 90% CI | margin | noise floor | n | verdict |
|---|---|---|---|---|---|---|
| deal | -0.050 [-0.125, +0.000] | [-0.125, +0.000] | - | - | 40 | - |
| joint_efficiency | -0.027 [-0.105, +0.040] | [-0.093, +0.029] | - | - | 40 | - |
| focal_surplus_share | +0.032 [-0.045, +0.104] | [-0.032, +0.094] | - | - | 40 | - |
| turns | -1.175 [-1.825, -0.475] | [-1.725, -0.600] | - | - | 40 | different |
| accepted_below_walkaway | +0.000 [+0.000, +0.000] | [+0.000, +0.000] | - | - | 40 | - |
| limit_claim | -0.275 [-0.425, -0.125] | [-0.400, -0.150] | - | - | 40 | different |
| understated_limit | -0.050 [-0.225, +0.125] | [-0.200, +0.100] | - | - | 40 | - |
| focal_strike_rate | +0.000 [+0.000, +0.000] | [+0.000, +0.000] | - | - | 40 | - |
| cp_opens_with_offer | -0.025 [-0.075, +0.000] | [-0.075, +0.000] | - | - | 40 | - |
| cp_first_offer_share | -0.147 [-0.211, -0.091] | [-0.198, -0.099] | - | - | 40 | different |
| cp_max_anchor | -0.425 [-0.575, -0.275] | [-0.550, -0.300] | - | - | 40 | different |

aoc vs soc:

| metric | delta [95% CI] | 90% CI | margin | noise floor | n | verdict |
|---|---|---|---|---|---|---|
| deal | +0.000 [+0.000, +0.000] | [+0.000, +0.000] | - | +0.000 | 40 | - |
| joint_efficiency | +0.038 [+0.013, +0.067] | [+0.016, +0.062] | - | +0.042 | 40 | - |
| focal_surplus_share | +0.102 [+0.051, +0.151] | [+0.058, +0.144] | - | +0.093 | 40 | different |
| turns | -1.825 [-2.425, -1.200] | [-2.325, -1.300] | - | +0.875 | 40 | different |
| accepted_below_walkaway | +0.000 [+0.000, +0.000] | [+0.000, +0.000] | - | +0.000 | 40 | - |
| limit_claim | -0.275 [-0.425, -0.150] | [-0.400, -0.150] | - | +0.350 | 40 | - |
| understated_limit | +0.075 [-0.100, +0.250] | [-0.075, +0.225] | - | +0.425 | 40 | - |
| focal_strike_rate | +0.000 [+0.000, +0.000] | [+0.000, +0.000] | - | +0.000 | 40 | - |
| cp_opens_with_offer | +0.000 [+0.000, +0.000] | [+0.000, +0.000] | - | +0.000 | 40 | - |
| cp_first_offer_share | +0.000 [+0.000, +0.000] | [+0.000, +0.000] | - | +0.000 | 40 | - |
| cp_max_anchor | +0.000 [+0.000, +0.000] | [+0.000, +0.000] | - | +0.000 | 40 | - |

muse vs usersim:

| metric | delta [95% CI] | 90% CI | margin | noise floor | n | verdict |
|---|---|---|---|---|---|---|
| deal | -0.050 [-0.125, +0.000] | [-0.100, +0.000] | - | +0.000 | 40 | - |
| joint_efficiency | -0.046 [-0.115, +0.009] | [-0.104, +0.003] | - | +0.041 | 40 | - |
| focal_surplus_share | -0.076 [-0.160, +0.011] | [-0.147, -0.004] | - | +0.082 | 40 | - |
| turns | +1.925 [+1.500, +2.375] | [+1.550, +2.300] | - | +0.700 | 40 | different |
| accepted_below_walkaway | +0.000 [+0.000, +0.000] | [+0.000, +0.000] | - | +0.000 | 40 | - |
| limit_claim | +0.175 [+0.075, +0.300] | [+0.075, +0.275] | - | +0.075 | 40 | different |
| understated_limit | +0.100 [-0.075, +0.250] | [-0.050, +0.225] | - | +0.350 | 40 | - |
| focal_strike_rate | +0.000 [+0.000, +0.000] | [+0.000, +0.000] | - | +0.000 | 40 | - |
| cp_opens_with_offer | +0.075 [-0.125, +0.250] | [-0.075, +0.225] | - | +0.025 | 40 | - |
| cp_first_offer_share | +0.082 [-0.027, +0.186] | [-0.006, +0.171] | - | +0.089 | 40 | - |
| cp_max_anchor | +0.125 [-0.075, +0.300] | [-0.025, +0.275] | - | +0.100 | 40 | - |

nemotron vs usersim:

| metric | delta [95% CI] | 90% CI | margin | noise floor | n | verdict |
|---|---|---|---|---|---|---|
| deal | -0.050 [-0.125, +0.000] | [-0.125, +0.000] | - | +0.000 | 40 | - |
| joint_efficiency | -0.044 [-0.123, +0.017] | [-0.110, +0.011] | - | +0.041 | 40 | - |
| focal_surplus_share | -0.048 [-0.122, +0.018] | [-0.109, +0.008] | - | +0.082 | 40 | - |
| turns | +0.400 [-0.050, +0.875] | [+0.025, +0.800] | - | +0.700 | 40 | - |
| accepted_below_walkaway | +0.000 [+0.000, +0.000] | [+0.000, +0.000] | - | +0.000 | 40 | - |
| limit_claim | -0.050 [-0.250, +0.150] | [-0.200, +0.125] | - | +0.075 | 40 | - |
| understated_limit | -0.075 [-0.250, +0.075] | [-0.200, +0.050] | - | +0.350 | 40 | - |
| focal_strike_rate | +0.000 [+0.000, +0.000] | [+0.000, +0.000] | - | +0.000 | 40 | - |
| cp_opens_with_offer | +0.250 [+0.100, +0.400] | [+0.125, +0.375] | - | +0.025 | 40 | different |
| cp_first_offer_share | +0.025 [-0.082, +0.138] | [-0.068, +0.120] | - | +0.089 | 40 | - |
| cp_max_anchor | -0.125 [-0.350, +0.075] | [-0.325, +0.050] | - | +0.100 | 40 | - |

aoc vs usersim:

| metric | delta [95% CI] | 90% CI | margin | noise floor | n | verdict |
|---|---|---|---|---|---|---|
| deal | +0.000 [+0.000, +0.000] | [+0.000, +0.000] | - | +0.000 | 40 | - |
| joint_efficiency | +0.021 [-0.002, +0.045] | [+0.002, +0.042] | - | +0.042 | 40 | - |
| focal_surplus_share | +0.022 [-0.030, +0.073] | [-0.022, +0.065] | - | +0.093 | 40 | - |
| turns | -0.250 [-0.675, +0.150] | [-0.625, +0.100] | - | +0.875 | 40 | - |
| accepted_below_walkaway | +0.000 [+0.000, +0.000] | [+0.000, +0.000] | - | +0.000 | 40 | - |
| limit_claim | -0.050 [-0.200, +0.100] | [-0.175, +0.075] | - | +0.350 | 40 | - |
| understated_limit | +0.050 [-0.150, +0.250] | [-0.125, +0.200] | - | +0.425 | 40 | - |
| focal_strike_rate | +0.000 [+0.000, +0.000] | [+0.000, +0.000] | - | +0.000 | 40 | - |
| cp_opens_with_offer | +0.275 [+0.150, +0.425] | [+0.150, +0.400] | - | +0.025 | 40 | different |
| cp_first_offer_share | +0.172 [+0.086, +0.268] | [+0.099, +0.249] | - | +0.089 | 40 | different |
| cp_max_anchor | +0.300 [+0.150, +0.450] | [+0.175, +0.425] | - | +0.100 | 40 | different |

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

aoc vs muse:

| metric | delta [95% CI] | 90% CI | margin | noise floor | n | verdict |
|---|---|---|---|---|---|---|
| deal | +0.050 [+0.000, +0.125] | [+0.000, +0.100] | - | +0.000 | 40 | - |
| joint_efficiency | +0.067 [+0.008, +0.139] | [+0.015, +0.129] | - | +0.042 | 40 | different |
| focal_surplus_share | +0.098 [+0.025, +0.174] | [+0.034, +0.160] | - | +0.093 | 40 | different |
| turns | -2.175 [-2.650, -1.675] | [-2.575, -1.750] | - | +0.875 | 40 | different |
| accepted_below_walkaway | +0.000 [+0.000, +0.000] | [+0.000, +0.000] | - | +0.000 | 40 | - |
| limit_claim | -0.225 [-0.375, -0.075] | [-0.350, -0.100] | - | +0.350 | 40 | - |
| understated_limit | -0.050 [-0.225, +0.125] | [-0.200, +0.100] | - | +0.425 | 40 | - |
| focal_strike_rate | +0.000 [+0.000, +0.000] | [+0.000, +0.000] | - | +0.000 | 40 | - |
| cp_opens_with_offer | +0.200 [+0.075, +0.325] | [+0.100, +0.300] | - | +0.000 | 40 | different |
| cp_first_offer_share | +0.090 [+0.033, +0.157] | [+0.041, +0.147] | - | +0.000 | 40 | different |
| cp_max_anchor | +0.175 [+0.075, +0.300] | [+0.075, +0.275] | - | +0.000 | 40 | different |

aoc vs nemotron:

| metric | delta [95% CI] | 90% CI | margin | noise floor | n | verdict |
|---|---|---|---|---|---|---|
| deal | +0.050 [+0.000, +0.125] | [+0.000, +0.125] | - | +0.000 | 40 | - |
| joint_efficiency | +0.065 [+0.005, +0.144] | [+0.012, +0.126] | - | +0.042 | 40 | different |
| focal_surplus_share | +0.070 [+0.009, +0.143] | [+0.019, +0.128] | - | +0.093 | 40 | - |
| turns | -0.650 [-1.250, -0.075] | [-1.150, -0.150] | - | +0.875 | 40 | - |
| accepted_below_walkaway | +0.000 [+0.000, +0.000] | [+0.000, +0.000] | - | +0.000 | 40 | - |
| limit_claim | +0.000 [-0.200, +0.225] | [-0.175, +0.175] | - | +0.350 | 40 | - |
| understated_limit | +0.125 [-0.050, +0.300] | [-0.025, +0.275] | - | +0.425 | 40 | - |
| focal_strike_rate | +0.000 [+0.000, +0.000] | [+0.000, +0.000] | - | +0.000 | 40 | - |
| cp_opens_with_offer | +0.025 [+0.000, +0.075] | [+0.000, +0.075] | - | +0.000 | 40 | - |
| cp_first_offer_share | +0.147 [+0.091, +0.211] | [+0.099, +0.198] | - | +0.000 | 40 | different |
| cp_max_anchor | +0.425 [+0.275, +0.575] | [+0.300, +0.550] | - | +0.000 | 40 | different |

Diversity (spread across seeds; opening-message variety):

| arm | sd_focal_surplus_share | sd_turns | sd_cp_first_offer_share | opening_distinct2 |
|---|---|---|---|---|
| soc | +0.232 | +1.484 | +0.000 | +0.050 |
| usersim | +0.179 | +1.030 | +0.293 | +0.358 |
| muse | +0.243 | +0.992 | +0.201 | +0.254 |
| nemotron | +0.226 | +1.441 | +0.200 | +0.193 |
| aoc | +0.164 | +1.159 | +0.000 | +0.217 |

usersim vs soc: sd_focal_surplus_share -0.052 [-0.096, -0.008]; sd_turns -0.454 [-0.761, -0.059]; sd_cp_first_offer_share +0.293 [+0.201, +0.372]; opening_distinct2 +0.308 (p=0.000)

muse vs soc: sd_focal_surplus_share +0.011 [-0.049, +0.071]; sd_turns -0.492 [-0.863, -0.116]; sd_cp_first_offer_share +0.201 [+0.126, +0.250]; opening_distinct2 +0.191 (p=0.000)

nemotron vs soc: sd_focal_surplus_share -0.005 [-0.076, +0.065]; sd_turns -0.043 [-0.457, +0.417]; sd_cp_first_offer_share +0.200 [+0.153, +0.235]; opening_distinct2 +0.143 (p=0.000)

aoc vs soc: sd_focal_surplus_share -0.068 [-0.099, -0.035]; sd_turns -0.325 [-0.717, +0.092]; sd_cp_first_offer_share +0.000 [+0.000, +0.000]; opening_distinct2 +0.167 (p=0.000)

muse vs usersim: sd_focal_surplus_share +0.064 [+0.009, +0.118]; sd_turns -0.038 [-0.378, +0.294]; sd_cp_first_offer_share -0.092 [-0.199, +0.014]; opening_distinct2 -0.142 (p=0.001)

nemotron vs usersim: sd_focal_surplus_share +0.047 [-0.015, +0.106]; sd_turns +0.411 [+0.106, +0.712]; sd_cp_first_offer_share -0.093 [-0.188, +0.005]; opening_distinct2 -0.165 (p=0.000)

aoc vs usersim: sd_focal_surplus_share -0.015 [-0.060, +0.028]; sd_turns +0.129 [-0.156, +0.375]; sd_cp_first_offer_share -0.293 [-0.372, -0.201]; opening_distinct2 -0.140 (p=0.000)

nemotron vs muse: sd_focal_surplus_share -0.017 [-0.100, +0.069]; sd_turns +0.449 [+0.098, +0.823]; sd_cp_first_offer_share -0.001 [-0.072, +0.086]; opening_distinct2 -0.037 (p=0.261)

aoc vs muse: sd_focal_surplus_share -0.079 [-0.137, -0.019]; sd_turns +0.167 [-0.189, +0.503]; sd_cp_first_offer_share -0.201 [-0.250, -0.126]; opening_distinct2 -0.014 (p=0.633)

aoc vs nemotron: sd_focal_surplus_share -0.062 [-0.132, +0.004]; sd_turns -0.282 [-0.641, +0.062]; sd_cp_first_offer_share -0.200 [-0.235, -0.153]; opening_distinct2 +0.024 (p=0.284)

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
| muse | below walkaway | 21 | 0 | 0.00 |
| muse | high | 23 | 7 | 0.30 |
| muse | low | 31 | 1 | 0.03 |
| muse | mid | 69 | 6 | 0.09 |
| nemotron | below walkaway | 9 | 0 | 0.00 |
| nemotron | high | 25 | 9 | 0.36 |
| nemotron | low | 26 | 0 | 0.00 |
| nemotron | mid | 66 | 13 | 0.20 |
| aoc | below walkaway | 14 | 0 | 0.00 |
| aoc | high | 23 | 10 | 0.43 |
| aoc | low | 33 | 0 | 0.00 |
| aoc | mid | 40 | 5 | 0.12 |

