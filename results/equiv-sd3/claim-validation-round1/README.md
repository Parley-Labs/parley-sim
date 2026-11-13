# Claim-rule validation, round 1

Rules: parley-sim claims.py as of commit 71fef17 ("v2"; daf0b5b and 330d10f in earlier versions of the history).
Sample: 150 agent sentences from equiv-sd3, stratified by the rules' label
(all 11 rule-flagged "understated", 64 "accurate", 75 "none"); run totals
524 accurate, 11 understated, 1541 none. One blind human labeller.

Result: kappa 0.63 (all labels), 0.50 (understated vs other; 0.07 reweighted to the pool). Fails the 0.84
bar. Precision: accurate 62/64, understated 7/11. Recall is the problem: 7 of
75 rule-"none" sentences were human-labelled understated (about 140 missed
bluffs across the run vs 7 caught; ~5% recall) and 26/75 contained some claim.

Follow-ups: definitions settled after this round ("up to X" and "best offer"
are ceiling claims; hedges like "a stretch" are implied inability; a sentence
naming two amounts is understated if either is within the limit). Rules are
revised against them and must be validated on sentences not used here.
