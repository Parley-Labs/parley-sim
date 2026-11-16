# equiv-sd3 (support_desk)

Cells: done=400
Spend: 5,477,709 input + 465,346 output tokens, $0.00 reported by provider
Verdict rule: margin (replicates averaged per seed; "different" needs a 95% interval excluding zero and a delta of at least the margin)

## opus

Means (replicate 0; episodes per arm: soc=40, usersim=40, aoc=40; comparisons use only seeds present in both arms, averaging each seed's replicates):

| metric | kind | soc | usersim | aoc |
|---|---|---|---|---|
| deal | relational | +1.000 | +1.000 | +1.000 |
| joint_efficiency | relational | +0.856 | +0.880 | +0.895 |
| focal_surplus_share | relational | +0.566 | +0.671 | +0.679 |
| turns | relational | +7.850 | +6.675 | +6.350 |
| accepted_below_walkaway | intrinsic | +0.000 | +0.000 | +0.000 |
| focal_strike_rate | covariate | +0.000 | +0.000 | +0.000 |
| cp_opens_with_offer | counterpart | +1.000 | +0.725 | +1.000 |
| cp_first_offer_share | counterpart | +1.000 | +0.851 | +1.000 |
| cp_max_anchor | counterpart | +1.000 | +0.725 | +1.000 |

usersim vs soc:

| metric | delta [95% CI] | 90% CI | margin | noise floor | n | verdict |
|---|---|---|---|---|---|---|
| deal | +0.000 [+0.000, +0.000] | [+0.000, +0.000] | - | +0.000 | 40 | - |
| joint_efficiency | +0.035 [+0.013, +0.060] | [+0.017, +0.055] | - | +0.052 | 40 | different |
| focal_surplus_share | +0.117 [+0.064, +0.169] | [+0.072, +0.160] | +0.080 | +0.106 | 40 | different |
| turns | -1.375 [-1.825, -0.887] | [-1.762, -0.975] | +0.800 | +0.800 | 40 | different |
| accepted_below_walkaway | +0.000 [+0.000, +0.000] | [+0.000, +0.000] | - | +0.000 | 40 | - |
| focal_strike_rate | +0.000 [+0.000, +0.000] | [+0.000, +0.000] | - | +0.000 | 40 | - |
| cp_opens_with_offer | -0.275 [-0.425, -0.150] | [-0.400, -0.150] | - | +0.000 | 40 | different |
| cp_first_offer_share | -0.154 [-0.235, -0.080] | [-0.220, -0.089] | - | +0.070 | 40 | different |
| cp_max_anchor | -0.287 [-0.425, -0.150] | [-0.400, -0.175] | - | +0.025 | 40 | different |

aoc vs soc:

| metric | delta [95% CI] | 90% CI | margin | noise floor | n | verdict |
|---|---|---|---|---|---|---|
| deal | +0.000 [+0.000, +0.000] | [+0.000, +0.000] | - | +0.000 | 40 | - |
| joint_efficiency | +0.045 [+0.018, +0.075] | [+0.022, +0.070] | - | +0.032 | 40 | different |
| focal_surplus_share | +0.121 [+0.061, +0.181] | [+0.068, +0.172] | +0.080 | +0.088 | 40 | different |
| turns | -1.600 [-2.000, -1.150] | [-1.950, -1.225] | +0.800 | +1.050 | 40 | different |
| accepted_below_walkaway | +0.000 [+0.000, +0.000] | [+0.000, +0.000] | - | +0.000 | 40 | - |
| focal_strike_rate | +0.000 [+0.000, +0.000] | [+0.000, +0.000] | - | +0.000 | 40 | - |
| cp_opens_with_offer | +0.000 [+0.000, +0.000] | [+0.000, +0.000] | - | +0.000 | 40 | - |
| cp_first_offer_share | +0.000 [+0.000, +0.000] | [+0.000, +0.000] | - | +0.000 | 40 | - |
| cp_max_anchor | +0.000 [+0.000, +0.000] | [+0.000, +0.000] | - | +0.000 | 40 | - |

aoc vs usersim:

| metric | delta [95% CI] | 90% CI | margin | noise floor | n | verdict |
|---|---|---|---|---|---|---|
| deal | +0.000 [+0.000, +0.000] | [+0.000, +0.000] | - | +0.000 | 40 | - |
| joint_efficiency | +0.010 [-0.004, +0.027] | [-0.002, +0.024] | - | +0.052 | 40 | - |
| focal_surplus_share | +0.004 [-0.024, +0.034] | [-0.020, +0.028] | +0.080 | +0.106 | 40 | equivalent |
| turns | -0.225 [-0.487, +0.037] | [-0.450, -0.013] | +0.800 | +1.050 | 40 | equivalent |
| accepted_below_walkaway | +0.000 [+0.000, +0.000] | [+0.000, +0.000] | - | +0.000 | 40 | - |
| focal_strike_rate | +0.000 [+0.000, +0.000] | [+0.000, +0.000] | - | +0.000 | 40 | - |
| cp_opens_with_offer | +0.275 [+0.150, +0.425] | [+0.150, +0.400] | - | +0.000 | 40 | different |
| cp_first_offer_share | +0.154 [+0.080, +0.235] | [+0.089, +0.220] | - | +0.070 | 40 | different |
| cp_max_anchor | +0.287 [+0.150, +0.425] | [+0.175, +0.400] | - | +0.025 | 40 | different |

Diversity (spread across seeds; opening-message variety):

| arm | sd_focal_surplus_share | sd_turns | sd_cp_first_offer_share | opening_distinct2 | n_openings |
|---|---|---|---|---|---|
| soc | +0.267 | +1.406 | +0.000 | +0.050 | 40 |
| usersim | +0.165 | +0.971 | +0.258 | +0.360 | 40 |
| aoc | +0.151 | +0.921 | +0.000 | +0.217 | 40 |

usersim vs soc: sd_focal_surplus_share -0.103 [-0.138, -0.060]; sd_turns -0.435 [-0.765, -0.052]; sd_cp_first_offer_share +0.258 [+0.177, +0.307]; opening_distinct2 +0.310 (p<0.001, n=40)

aoc vs soc: sd_focal_surplus_share -0.116 [-0.155, -0.074]; sd_turns -0.485 [-0.798, -0.122]; sd_cp_first_offer_share +0.000 [+0.000, +0.000]; opening_distinct2 +0.167 (p<0.001, n=40)

aoc vs usersim: sd_focal_surplus_share -0.014 [-0.046, +0.020]; sd_turns -0.050 [-0.309, +0.212]; sd_cp_first_offer_share -0.258 [-0.307, -0.177]; opening_distinct2 -0.143 (p<0.001, n=40)

Acceptance by offer quality (every focal decision taken facing an offer it could legally accept, declines included; an offer that stands for several decisions counts at each):

| arm | bin | decisions facing an offer | accepted | rate |
|---|---|---|---|---|
| soc | below walkaway | 5 | 0 | 0.00 |
| soc | high | 7 | 1 | 0.14 |
| soc | low | 8 | 1 | 0.12 |
| soc | mid | 18 | 4 | 0.22 |
| usersim | below walkaway | 1 | 0 | 0.00 |
| usersim | high | 17 | 5 | 0.29 |
| usersim | low | 6 | 0 | 0.00 |
| usersim | mid | 39 | 8 | 0.21 |
| aoc | below walkaway | 1 | 0 | 0.00 |
| aoc | high | 16 | 8 | 0.50 |
| aoc | low | 7 | 0 | 0.00 |
| aoc | mid | 41 | 6 | 0.15 |

Experimental metrics. _(experimental) metrics come from rules not yet validated against human labels well enough to quote as rates._

| metric | kind | soc | usersim | aoc |
|---|---|---|---|---|
| limit_claim | intrinsic (experimental) | +0.900 | +0.775 | +0.875 |
| understated_limit | intrinsic (experimental) | +0.150 | +0.125 | +0.175 |

usersim vs soc:

| metric | delta [95% CI] | 90% CI | margin | noise floor | n | verdict |
|---|---|---|---|---|---|---|
| limit_claim | -0.075 [-0.163, +0.013] | [-0.150, +0.000] | - | +0.150 | 40 | - |
| understated_limit | -0.013 [-0.138, +0.113] | [-0.113, +0.087] | - | +0.275 | 40 | - |

aoc vs soc:

| metric | delta [95% CI] | 90% CI | margin | noise floor | n | verdict |
|---|---|---|---|---|---|---|
| limit_claim | -0.025 [-0.100, +0.062] | [-0.087, +0.050] | - | +0.100 | 40 | - |
| understated_limit | +0.013 [-0.138, +0.150] | [-0.100, +0.125] | - | +0.325 | 40 | - |

aoc vs usersim:

| metric | delta [95% CI] | 90% CI | margin | noise floor | n | verdict |
|---|---|---|---|---|---|---|
| limit_claim | +0.050 [-0.037, +0.150] | [-0.025, +0.125] | - | +0.150 | 40 | - |
| understated_limit | +0.025 [-0.075, +0.125] | [-0.062, +0.113] | - | +0.325 | 40 | - |

## sonnet

Means (replicate 0; episodes per arm: soc=40, usersim=40, aoc=40; comparisons use only seeds present in both arms, averaging each seed's replicates):

| metric | kind | soc | usersim | aoc |
|---|---|---|---|---|
| deal | relational | +1.000 | +1.000 | +1.000 |
| joint_efficiency | relational | +0.853 | +0.871 | +0.892 |
| focal_surplus_share | relational | +0.611 | +0.691 | +0.713 |
| turns | relational | +7.950 | +6.375 | +6.125 |
| accepted_below_walkaway | intrinsic | +0.000 | +0.000 | +0.000 |
| focal_strike_rate | covariate | +0.000 | +0.000 | +0.000 |
| cp_opens_with_offer | counterpart | +1.000 | +0.725 | +1.000 |
| cp_first_offer_share | counterpart | +1.000 | +0.828 | +1.000 |
| cp_max_anchor | counterpart | +1.000 | +0.700 | +1.000 |

usersim vs soc:

| metric | delta [95% CI] | 90% CI | margin | noise floor | n | verdict |
|---|---|---|---|---|---|---|
| deal | +0.000 [+0.000, +0.000] | [+0.000, +0.000] | - | +0.000 | 40 | - |
| joint_efficiency | +0.018 [-0.004, +0.040] | [-0.001, +0.036] | - | +0.041 | 40 | - |
| focal_surplus_share | +0.077 [+0.024, +0.128] | [+0.033, +0.119] | +0.080 | +0.082 | 40 | inconclusive |
| turns | -1.600 [-2.038, -1.125] | [-1.975, -1.188] | +0.800 | +0.700 | 40 | different |
| accepted_below_walkaway | +0.000 [+0.000, +0.000] | [+0.000, +0.000] | - | +0.000 | 40 | - |
| focal_strike_rate | +0.000 [+0.000, +0.000] | [+0.000, +0.000] | - | +0.000 | 40 | - |
| cp_opens_with_offer | -0.263 [-0.400, -0.125] | [-0.375, -0.150] | - | +0.025 | 40 | different |
| cp_first_offer_share | -0.145 [-0.227, -0.071] | [-0.213, -0.082] | - | +0.089 | 40 | different |
| cp_max_anchor | -0.250 [-0.375, -0.125] | [-0.350, -0.150] | - | +0.100 | 40 | different |

aoc vs soc:

| metric | delta [95% CI] | 90% CI | margin | noise floor | n | verdict |
|---|---|---|---|---|---|---|
| deal | +0.000 [+0.000, +0.000] | [+0.000, +0.000] | - | +0.000 | 40 | - |
| joint_efficiency | +0.037 [+0.014, +0.063] | [+0.017, +0.058] | - | +0.042 | 40 | different |
| focal_surplus_share | +0.083 [+0.036, +0.133] | [+0.042, +0.125] | +0.080 | +0.093 | 40 | different |
| turns | -1.738 [-2.250, -1.188] | [-2.163, -1.275] | +0.800 | +0.875 | 40 | different |
| accepted_below_walkaway | +0.000 [+0.000, +0.000] | [+0.000, +0.000] | - | +0.000 | 40 | - |
| focal_strike_rate | +0.000 [+0.000, +0.000] | [+0.000, +0.000] | - | +0.000 | 40 | - |
| cp_opens_with_offer | +0.000 [+0.000, +0.000] | [+0.000, +0.000] | - | +0.000 | 40 | - |
| cp_first_offer_share | +0.000 [+0.000, +0.000] | [+0.000, +0.000] | - | +0.000 | 40 | - |
| cp_max_anchor | +0.000 [+0.000, +0.000] | [+0.000, +0.000] | - | +0.000 | 40 | - |

aoc vs usersim:

| metric | delta [95% CI] | 90% CI | margin | noise floor | n | verdict |
|---|---|---|---|---|---|---|
| deal | +0.000 [+0.000, +0.000] | [+0.000, +0.000] | - | +0.000 | 40 | - |
| joint_efficiency | +0.020 [+0.002, +0.039] | [+0.005, +0.036] | - | +0.042 | 40 | different |
| focal_surplus_share | +0.006 [-0.028, +0.040] | [-0.023, +0.035] | +0.080 | +0.093 | 40 | equivalent |
| turns | -0.138 [-0.450, +0.175] | [-0.400, +0.125] | +0.800 | +0.875 | 40 | equivalent |
| accepted_below_walkaway | +0.000 [+0.000, +0.000] | [+0.000, +0.000] | - | +0.000 | 40 | - |
| focal_strike_rate | +0.000 [+0.000, +0.000] | [+0.000, +0.000] | - | +0.000 | 40 | - |
| cp_opens_with_offer | +0.263 [+0.125, +0.400] | [+0.150, +0.375] | - | +0.025 | 40 | different |
| cp_first_offer_share | +0.145 [+0.071, +0.227] | [+0.082, +0.213] | - | +0.089 | 40 | different |
| cp_max_anchor | +0.250 [+0.125, +0.375] | [+0.150, +0.350] | - | +0.100 | 40 | different |

Diversity (spread across seeds; opening-message variety):

| arm | sd_focal_surplus_share | sd_turns | sd_cp_first_offer_share | opening_distinct2 | n_openings |
|---|---|---|---|---|---|
| soc | +0.232 | +1.484 | +0.000 | +0.050 | 40 |
| usersim | +0.179 | +1.030 | +0.293 | +0.358 | 40 |
| aoc | +0.164 | +1.159 | +0.000 | +0.217 | 40 |

usersim vs soc: sd_focal_surplus_share -0.052 [-0.096, -0.008]; sd_turns -0.454 [-0.761, -0.059]; sd_cp_first_offer_share +0.293 [+0.201, +0.372]; opening_distinct2 +0.308 (p<0.001, n=40)

aoc vs soc: sd_focal_surplus_share -0.068 [-0.099, -0.035]; sd_turns -0.325 [-0.717, +0.092]; sd_cp_first_offer_share +0.000 [+0.000, +0.000]; opening_distinct2 +0.167 (p<0.001, n=40)

aoc vs usersim: sd_focal_surplus_share -0.015 [-0.060, +0.028]; sd_turns +0.129 [-0.156, +0.375]; sd_cp_first_offer_share -0.293 [-0.372, -0.201]; opening_distinct2 -0.140 (p<0.001, n=40)

Acceptance by offer quality (every focal decision taken facing an offer it could legally accept, declines included; an offer that stands for several decisions counts at each):

| arm | bin | decisions facing an offer | accepted | rate |
|---|---|---|---|---|
| soc | below walkaway | 5 | 0 | 0.00 |
| soc | high | 9 | 3 | 0.33 |
| soc | low | 8 | 0 | 0.00 |
| soc | mid | 19 | 3 | 0.16 |
| usersim | below walkaway | 1 | 0 | 0.00 |
| usersim | high | 17 | 5 | 0.29 |
| usersim | low | 6 | 0 | 0.00 |
| usersim | mid | 31 | 8 | 0.26 |
| aoc | below walkaway | 1 | 0 | 0.00 |
| aoc | high | 19 | 10 | 0.53 |
| aoc | low | 11 | 0 | 0.00 |
| aoc | mid | 28 | 5 | 0.18 |

Experimental metrics. _(experimental) metrics come from rules not yet validated against human labels well enough to quote as rates._

| metric | kind | soc | usersim | aoc |
|---|---|---|---|---|
| limit_claim | intrinsic (experimental) | +0.975 | +0.750 | +0.700 |
| understated_limit | intrinsic (experimental) | +0.275 | +0.300 | +0.350 |

usersim vs soc:

| metric | delta [95% CI] | 90% CI | margin | noise floor | n | verdict |
|---|---|---|---|---|---|---|
| limit_claim | -0.212 [-0.350, -0.087] | [-0.325, -0.100] | - | +0.075 | 40 | different |
| understated_limit | +0.025 [-0.138, +0.175] | [-0.113, +0.150] | - | +0.350 | 40 | - |

aoc vs soc:

| metric | delta [95% CI] | 90% CI | margin | noise floor | n | verdict |
|---|---|---|---|---|---|---|
| limit_claim | -0.225 [-0.338, -0.125] | [-0.312, -0.138] | - | +0.350 | 40 | different |
| understated_limit | +0.037 [-0.100, +0.188] | [-0.087, +0.163] | - | +0.425 | 40 | - |

aoc vs usersim:

| metric | delta [95% CI] | 90% CI | margin | noise floor | n | verdict |
|---|---|---|---|---|---|---|
| limit_claim | -0.013 [-0.113, +0.087] | [-0.100, +0.075] | - | +0.350 | 40 | - |
| understated_limit | +0.013 [-0.125, +0.138] | [-0.100, +0.113] | - | +0.425 | 40 | - |

