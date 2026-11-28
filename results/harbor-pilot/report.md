# harbor-pilot-harbor (support_desk)

Cells: done=400
Spend: 3,543,553 input + 199,187 output tokens, $0.00 reported by provider
Verdict rule: margin (replicates averaged per seed; "different" needs a 95% interval excluding zero and, where a margin is set, a delta of at least the margin in either direction)

## claude-code.anthropic-claude-sonnet-5-5

Means (replicate 0; episodes per arm: soc=40, usersim=40, aoc=40; comparisons use only seeds present in both arms, averaging each seed's replicates):

| metric | kind | soc | usersim | aoc |
|---|---|---|---|---|
| deal | relational | +0.900 | +0.975 | +1.000 |
| joint_efficiency | relational | +0.794 | +0.852 | +0.886 |
| focal_surplus_share | relational | +0.565 | +0.751 | +0.783 |
| turns | relational | +8.425 | +7.150 | +6.650 |
| accepted_below_walkaway | intrinsic | +0.000 | +0.000 | +0.000 |
| focal_strike_rate | covariate | +0.000 | +0.000 | +0.000 |
| cp_opens_with_offer | counterpart | +1.000 | +0.425 | +1.000 |
| cp_first_offer_share | counterpart | +1.000 | +0.627 | +0.996 |
| cp_max_anchor | counterpart | +1.000 | +0.400 | +0.975 |

usersim vs soc:

| metric | delta [95% CI] | 90% CI | margin | noise floor | n | verdict |
|---|---|---|---|---|---|---|
| deal | +0.087 [+0.000, +0.200] | [+0.013, +0.175] | - | +0.025 | 40 | - |
| joint_efficiency | +0.083 [-0.002, +0.192] | [+0.008, +0.172] | - | +0.084 | 40 | - |
| focal_surplus_share | +0.203 [+0.103, +0.317] | [+0.116, +0.299] | +0.080 | +0.112 | 40 | different |
| turns | -1.325 [-1.725, -0.900] | [-1.675, -0.975] | +0.800 | +0.650 | 40 | different |
| accepted_below_walkaway | +0.000 [+0.000, +0.000] | [+0.000, +0.000] | - | +0.000 | 40 | - |
| focal_strike_rate | +0.000 [+0.000, +0.000] | [+0.000, +0.000] | - | +0.000 | 40 | - |
| cp_opens_with_offer | -0.600 [-0.738, -0.450] | [-0.725, -0.475] | - | +0.050 | 40 | different |
| cp_first_offer_share | -0.384 [-0.475, -0.287] | [-0.462, -0.304] | - | +0.156 | 40 | different |
| cp_max_anchor | -0.625 [-0.750, -0.487] | [-0.738, -0.512] | - | +0.200 | 40 | different |

aoc vs soc:

| metric | delta [95% CI] | 90% CI | margin | noise floor | n | verdict |
|---|---|---|---|---|---|---|
| deal | +0.100 [+0.025, +0.200] | [+0.025, +0.200] | - | +0.000 | 40 | different |
| joint_efficiency | +0.096 [+0.014, +0.203] | [+0.025, +0.185] | - | +0.052 | 40 | different |
| focal_surplus_share | +0.209 [+0.105, +0.322] | [+0.122, +0.303] | +0.080 | +0.089 | 40 | different |
| turns | -1.962 [-2.388, -1.525] | [-2.325, -1.587] | +0.800 | +0.775 | 40 | different |
| accepted_below_walkaway | +0.000 [+0.000, +0.000] | [+0.000, +0.000] | - | +0.000 | 40 | - |
| focal_strike_rate | +0.000 [+0.000, +0.000] | [+0.000, +0.000] | - | +0.000 | 40 | - |
| cp_opens_with_offer | +0.000 [+0.000, +0.000] | [+0.000, +0.000] | - | +0.000 | 40 | - |
| cp_first_offer_share | -0.002 [-0.006, +0.000] | [-0.006, +0.000] | - | +0.004 | 40 | - |
| cp_max_anchor | -0.013 [-0.037, +0.000] | [-0.037, +0.000] | - | +0.025 | 40 | - |

aoc vs usersim:

| metric | delta [95% CI] | 90% CI | margin | noise floor | n | verdict |
|---|---|---|---|---|---|---|
| deal | +0.013 [+0.000, +0.037] | [+0.000, +0.037] | - | +0.025 | 40 | - |
| joint_efficiency | +0.014 [-0.012, +0.042] | [-0.008, +0.037] | - | +0.084 | 40 | - |
| focal_surplus_share | +0.006 [-0.032, +0.051] | [-0.027, +0.043] | +0.080 | +0.112 | 40 | equivalent |
| turns | -0.637 [-0.988, -0.300] | [-0.925, -0.362] | +0.800 | +0.775 | 40 | inconclusive |
| accepted_below_walkaway | +0.000 [+0.000, +0.000] | [+0.000, +0.000] | - | +0.000 | 40 | - |
| focal_strike_rate | +0.000 [+0.000, +0.000] | [+0.000, +0.000] | - | +0.000 | 40 | - |
| cp_opens_with_offer | +0.600 [+0.450, +0.738] | [+0.475, +0.725] | - | +0.050 | 40 | different |
| cp_first_offer_share | +0.382 [+0.285, +0.473] | [+0.302, +0.459] | - | +0.156 | 40 | different |
| cp_max_anchor | +0.613 [+0.475, +0.738] | [+0.500, +0.725] | - | +0.200 | 40 | different |

Diversity (spread across seeds; opening-message variety):

| arm | sd_focal_surplus_share | sd_turns | sd_cp_first_offer_share | opening_distinct2 | n_openings |
|---|---|---|---|---|---|
| soc | +0.306 | +1.318 | +0.000 | +0.050 | 40 |
| usersim | +0.216 | +0.864 | +0.337 | +0.369 | 40 |
| aoc | +0.165 | +0.975 | +0.026 | +0.148 | 40 |

usersim vs soc: sd_focal_surplus_share -0.090 [-0.159, -0.023]; sd_turns -0.454 [-0.906, +0.058]; sd_cp_first_offer_share +0.337 [+0.293, +0.376]; opening_distinct2 +0.319 (p<0.001, n=40)

aoc vs soc: sd_focal_surplus_share -0.141 [-0.196, -0.075]; sd_turns -0.343 [-0.701, +0.086]; sd_cp_first_offer_share +0.026 [+0.000, +0.044]; opening_distinct2 +0.098 (p<0.001, n=40)

aoc vs usersim: sd_focal_surplus_share -0.051 [-0.122, +0.012]; sd_turns +0.112 [-0.183, +0.433]; sd_cp_first_offer_share -0.311 [-0.366, -0.261]; opening_distinct2 -0.221 (p<0.001, n=40)

Acceptance by offer quality (every focal decision taken facing an offer it could legally accept, declines included; an offer that stands for several decisions counts at each):

| arm | bin | decisions facing an offer | accepted | rate |
|---|---|---|---|---|
| soc | below walkaway | 7 | 0 | 0.00 |
| soc | high | 17 | 6 | 0.35 |
| soc | low | 6 | 0 | 0.00 |
| soc | mid | 17 | 3 | 0.18 |
| usersim | below walkaway | 1 | 0 | 0.00 |
| usersim | high | 35 | 9 | 0.26 |
| usersim | low | 8 | 0 | 0.00 |
| usersim | mid | 25 | 1 | 0.04 |
| aoc | below walkaway | 2 | 0 | 0.00 |
| aoc | high | 27 | 9 | 0.33 |
| aoc | low | 6 | 0 | 0.00 |
| aoc | mid | 27 | 1 | 0.04 |

Experimental metrics. _(experimental) metrics come from rules not yet validated against human labels well enough to quote as rates._

| metric | kind | soc | usersim | aoc |
|---|---|---|---|---|
| limit_claim | intrinsic (experimental) | +0.625 | +0.425 | +0.475 |
| understated_limit | intrinsic (experimental) | +0.150 | +0.075 | +0.125 |

usersim vs soc:

| metric | delta [95% CI] | 90% CI | margin | noise floor | n | verdict |
|---|---|---|---|---|---|---|
| limit_claim | -0.163 [-0.338, +0.025] | [-0.312, -0.013] | - | +0.375 | 40 | - |
| understated_limit | -0.050 [-0.175, +0.075] | [-0.163, +0.050] | - | +0.200 | 40 | - |

aoc vs soc:

| metric | delta [95% CI] | 90% CI | margin | noise floor | n | verdict |
|---|---|---|---|---|---|---|
| limit_claim | -0.212 [-0.362, -0.075] | [-0.338, -0.087] | - | +0.375 | 40 | different |
| understated_limit | -0.062 [-0.188, +0.050] | [-0.163, +0.025] | - | +0.175 | 40 | - |

aoc vs usersim:

| metric | delta [95% CI] | 90% CI | margin | noise floor | n | verdict |
|---|---|---|---|---|---|---|
| limit_claim | -0.050 [-0.200, +0.087] | [-0.175, +0.075] | - | +0.375 | 40 | - |
| understated_limit | -0.013 [-0.113, +0.075] | [-0.087, +0.062] | - | +0.200 | 40 | - |

## terminus-2.anthropic-claude-sonnet-5-5

Means (replicate 0; episodes per arm: soc=40, usersim=40, aoc=40; comparisons use only seeds present in both arms, averaging each seed's replicates):

| metric | kind | soc | usersim | aoc |
|---|---|---|---|---|
| deal | relational | +0.925 | +1.000 | +0.975 |
| joint_efficiency | relational | +0.812 | +0.873 | +0.827 |
| focal_surplus_share | relational | +0.603 | +0.806 | +0.812 |
| turns | relational | +8.775 | +7.275 | +7.225 |
| accepted_below_walkaway | intrinsic | +0.000 | +0.000 | +0.000 |
| focal_strike_rate | covariate | +0.008 | +0.035 | +0.015 |
| cp_opens_with_offer | counterpart | +1.000 | +0.425 | +1.000 |
| cp_first_offer_share | counterpart | +1.000 | +0.594 | +1.000 |
| cp_max_anchor | counterpart | +1.000 | +0.350 | +1.000 |

usersim vs soc:

| metric | delta [95% CI] | 90% CI | margin | noise floor | n | verdict |
|---|---|---|---|---|---|---|
| deal | +0.075 [+0.000, +0.175] | [+0.025, +0.150] | - | +0.000 | 40 | - |
| joint_efficiency | +0.065 [-0.012, +0.154] | [-0.000, +0.140] | - | +0.043 | 40 | - |
| focal_surplus_share | +0.212 [+0.121, +0.312] | [+0.132, +0.296] | +0.080 | +0.096 | 40 | different |
| turns | -1.600 [-1.938, -1.262] | [-1.887, -1.312] | +0.800 | +0.700 | 40 | different |
| accepted_below_walkaway | +0.000 [+0.000, +0.000] | [+0.000, +0.000] | - | +0.000 | 40 | - |
| focal_strike_rate | +0.013 [-0.014, +0.042] | [-0.008, +0.036] | - | +0.042 | 40 | - |
| cp_opens_with_offer | -0.575 [-0.725, -0.425] | [-0.700, -0.438] | - | +0.050 | 40 | different |
| cp_first_offer_share | -0.407 [-0.509, -0.305] | [-0.494, -0.320] | - | +0.152 | 40 | different |
| cp_max_anchor | -0.637 [-0.775, -0.500] | [-0.762, -0.525] | - | +0.125 | 40 | different |

aoc vs soc:

| metric | delta [95% CI] | 90% CI | margin | noise floor | n | verdict |
|---|---|---|---|---|---|---|
| deal | +0.062 [-0.013, +0.150] | [+0.000, +0.138] | - | +0.025 | 40 | - |
| joint_efficiency | +0.047 [-0.028, +0.134] | [-0.019, +0.119] | - | +0.090 | 40 | - |
| focal_surplus_share | +0.232 [+0.130, +0.349] | [+0.143, +0.330] | +0.080 | +0.116 | 40 | different |
| turns | -1.750 [-2.075, -1.425] | [-2.038, -1.475] | +0.800 | +0.850 | 40 | different |
| accepted_below_walkaway | +0.000 [+0.000, +0.000] | [+0.000, +0.000] | - | +0.000 | 40 | - |
| focal_strike_rate | -0.001 [-0.013, +0.009] | [-0.009, +0.006] | - | +0.015 | 40 | - |
| cp_opens_with_offer | +0.000 [+0.000, +0.000] | [+0.000, +0.000] | - | +0.000 | 40 | - |
| cp_first_offer_share | +0.000 [+0.000, +0.000] | [+0.000, +0.000] | - | +0.000 | 40 | - |
| cp_max_anchor | +0.000 [+0.000, +0.000] | [+0.000, +0.000] | - | +0.000 | 40 | - |

aoc vs usersim:

| metric | delta [95% CI] | 90% CI | margin | noise floor | n | verdict |
|---|---|---|---|---|---|---|
| deal | -0.013 [-0.037, +0.000] | [-0.037, +0.000] | - | +0.025 | 40 | - |
| joint_efficiency | -0.018 [-0.057, +0.014] | [-0.050, +0.009] | - | +0.090 | 40 | - |
| focal_surplus_share | +0.020 [-0.031, +0.069] | [-0.020, +0.061] | +0.080 | +0.116 | 40 | equivalent |
| turns | -0.150 [-0.463, +0.138] | [-0.400, +0.087] | +0.800 | +0.850 | 40 | equivalent |
| accepted_below_walkaway | +0.000 [+0.000, +0.000] | [+0.000, +0.000] | - | +0.000 | 40 | - |
| focal_strike_rate | -0.014 [-0.041, +0.008] | [-0.036, +0.004] | - | +0.042 | 40 | - |
| cp_opens_with_offer | +0.575 [+0.425, +0.725] | [+0.438, +0.700] | - | +0.050 | 40 | different |
| cp_first_offer_share | +0.407 [+0.305, +0.509] | [+0.320, +0.494] | - | +0.152 | 40 | different |
| cp_max_anchor | +0.637 [+0.500, +0.775] | [+0.525, +0.762] | - | +0.125 | 40 | different |

Diversity (spread across seeds; opening-message variety):

| arm | sd_focal_surplus_share | sd_turns | sd_cp_first_offer_share | opening_distinct2 | n_openings |
|---|---|---|---|---|---|
| soc | +0.291 | +0.891 | +0.000 | +0.050 | 40 |
| usersim | +0.195 | +0.816 | +0.345 | +0.328 | 40 |
| aoc | +0.211 | +0.974 | +0.000 | +0.178 | 40 |

usersim vs soc: sd_focal_surplus_share -0.096 [-0.149, -0.037]; sd_turns -0.075 [-0.317, +0.189]; sd_cp_first_offer_share +0.345 [+0.288, +0.392]; opening_distinct2 +0.278 (p<0.001, n=40)

aoc vs soc: sd_focal_surplus_share -0.079 [-0.165, +0.014]; sd_turns +0.082 [-0.229, +0.394]; sd_cp_first_offer_share +0.000 [+0.000, +0.000]; opening_distinct2 +0.128 (p<0.001, n=40)

aoc vs usersim: sd_focal_surplus_share +0.016 [-0.060, +0.099]; sd_turns +0.158 [-0.182, +0.465]; sd_cp_first_offer_share -0.345 [-0.392, -0.288]; opening_distinct2 -0.150 (p<0.001, n=40)

Acceptance by offer quality (every focal decision taken facing an offer it could legally accept, declines included; an offer that stands for several decisions counts at each):

| arm | bin | decisions facing an offer | accepted | rate |
|---|---|---|---|---|
| soc | below walkaway | 11 | 0 | 0.00 |
| soc | high | 22 | 9 | 0.41 |
| soc | low | 6 | 0 | 0.00 |
| soc | mid | 15 | 1 | 0.07 |
| usersim | below walkaway | 1 | 0 | 0.00 |
| usersim | high | 48 | 16 | 0.33 |
| usersim | low | 9 | 0 | 0.00 |
| usersim | mid | 23 | 1 | 0.04 |
| aoc | below walkaway | 2 | 0 | 0.00 |
| aoc | high | 45 | 7 | 0.16 |
| aoc | low | 8 | 0 | 0.00 |
| aoc | mid | 21 | 0 | 0.00 |

Experimental metrics. _(experimental) metrics come from rules not yet validated against human labels well enough to quote as rates._

| metric | kind | soc | usersim | aoc |
|---|---|---|---|---|
| limit_claim | intrinsic (experimental) | +0.825 | +0.500 | +0.675 |
| understated_limit | intrinsic (experimental) | +0.250 | +0.175 | +0.125 |

usersim vs soc:

| metric | delta [95% CI] | 90% CI | margin | noise floor | n | verdict |
|---|---|---|---|---|---|---|
| limit_claim | -0.275 [-0.425, -0.138] | [-0.400, -0.150] | - | +0.300 | 40 | different |
| understated_limit | -0.037 [-0.188, +0.113] | [-0.163, +0.087] | - | +0.275 | 40 | - |

aoc vs soc:

| metric | delta [95% CI] | 90% CI | margin | noise floor | n | verdict |
|---|---|---|---|---|---|---|
| limit_claim | -0.075 [-0.188, +0.025] | [-0.163, +0.013] | - | +0.300 | 40 | - |
| understated_limit | -0.100 [-0.250, +0.037] | [-0.225, +0.013] | - | +0.250 | 40 | - |

aoc vs usersim:

| metric | delta [95% CI] | 90% CI | margin | noise floor | n | verdict |
|---|---|---|---|---|---|---|
| limit_claim | +0.200 [+0.050, +0.350] | [+0.062, +0.325] | - | +0.300 | 40 | different |
| understated_limit | -0.062 [-0.188, +0.050] | [-0.163, +0.037] | - | +0.275 | 40 | - |

