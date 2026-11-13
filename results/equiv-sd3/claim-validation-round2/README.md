# Claim-rule validation, round 2 (held-out)

Rules: parley-sim claims.py at commit 7bc41cf ("v3"; 8367fcf and 06dbb6c in earlier versions of the history), frozen before this
sample was drawn or labelled. Sample: 150 agent sentences from equiv-sd3, none
used in round 1, stratified by the rules' label (36 understated, 3
overstated, 36 accurate, 75 none; run totals 94 / 3 / 507 / 1446). One blind
human labeller, guide revised with the definitions settled after round 1.

Result: kappa 0.66 (all labels), 0.64 (understated vs other); bar 0.84, not met.
Precision: accurate 36/36, understated 23/36, overstated 0/3. Reweighted to
the held-out pool (the run minus round 1's 150 sentences), bluff recall is
about 38%, 95% interval (0.23, 0.75) (round 1: about 5%).
Kappa above is on the stratified sample, which over-represents flagged
sentences. Reweighted to the pool the sample was drawn from (`parley-sim
kappa`, `weighted_to_pool` in `kappa.json`), bluff kappa is 0.45, 95%
interval (0.28, 0.68) (round 1: 0.07); precision stays 64%, 95% interval
(0.47, 0.78). The intervals are from a stratified bootstrap.

Errors are mostly systematic scope: an issue-specific frame ("my maximum
discount: 20% off, a $50 credit") attaching amounts of the other issue, two
claims in one sentence merging (", and $25 is the most I can offer"), and an
"up to" running past "and I can". The rest are ambiguous boundary cases
(hedge followed by granting; billing-conditioned package refusals). No second
labeller, so labeller noise cannot be separated from rule error.
