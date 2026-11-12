# Reviewing parley-sim

Thank you for reviewing this. This guide says what parley-sim is, what we
claim, how to check each claim yourself, and where we already know the
weak points are. We'd rather hear about a problem now than after release, so
please be adversarial.

## What it is, in one paragraph

parley-sim evaluates an LLM agent against counterparts that adapt to it. The
same scenarios (same seeds) are played against a scripted counterpart and
against one or more LLM counterparts; a report compares the arms within
matched pairs, with bootstrap intervals, a replay-based noise floor, and
optional pre-registered equivalence margins. Metrics are computed from the
environment's exact ground truth, not by an LLM judge. It ships one small demo
environment (`support_desk`, a retention negotiation), a tested example of a
custom environment (`examples/haggle.py`), and adapters for NeMo Gym and
OpenEnv.

## Where to leave comments

Comment on any line of the framework in the review-only pull request
[#2](https://github.com/Parley-Labs/parley-sim/pull/2): every file of code,
tests, docs and configs shows there as added, so nothing is limited to what
changed recently. [#1](https://github.com/Parley-Labs/parley-sim/pull/1) holds
the same code and is the one that will be merged; #2 is never merged.

## Setup (about 5 minutes, no API keys)

```bash
git clone https://github.com/Parley-Labs/parley-sim && cd parley-sim
uv sync --extra openai --extra anthropic --extra openenv
uv run pytest -q                       # about 180 tests; 2 skip without the gym and usersim extras
uv run parley-sim smoke                # whole pipeline offline; prints "replay check: ok"
PYTHONPATH=examples uv run parley-sim matched examples/haggle.yaml --out /tmp/haggle
```

Nothing below needs a key. Re-running the experiments themselves does (about
$25 of model calls for `equiv-sd3` at list prices), and the persona arm also
needs NVIDIA's persona dataset (see `docs/personas.md`).

## Where to start reading

| If you care about | Read |
| --- | --- |
| The method and the statistics | `docs/concepts.md`, then `src/parley_sim/metrics/__init__.py` |
| Whether the results hold | `results/README.md`, `configs/equiv-sd3.yaml` (header = the pre-registered plan), `results/equiv-sd3/report.md` |
| The runtime and replay | `src/parley_sim/runtime/episode.py`, `src/parley_sim/matched/__init__.py` |
| The demo environment | `src/parley_sim/envs/support_desk/` |
| Plugging in your own environment | `docs/custom-environments.md`, `examples/haggle.py` |
| Training-stack integration | `docs/adapters.md`, `src/parley_sim/adapters/` |
| The experimental honesty metric | `docs/claims.md`, `src/parley_sim/envs/support_desk/claims.py` |

## The claims, and how to check each

1. **Matched pairs isolate the counterpart.** Within a `pair_key`, only the
   counterpart differs. Check `build_grid` in `matched/__init__.py` and the
   scenario-fingerprint check in `build_report`.
2. **Traces are exactly replayable.**
   `uv run parley-sim verify results/*/traces/*.jsonl` replays all 400
   published traces and checks every state hash, the ground truth and the
   outcomes.
3. **The published reports are reproducible from the published data.**
   `uv run parley-sim report results/equiv-sd3 --margins focal_surplus_share=0.08,turns=0.8`
   and compare with `results/equiv-sd3/report.md`. Known, documented
   differences are listed at the end of `results/README.md`.
4. **H1 (pre-registered): plain-LLM and persona customers give equivalent
   agent outcomes** on surplus share and turns. Check the "aoc vs usersim"
   tables in `results/equiv-sd3/report.md` against the rule in the config
   header. The persona arm's traces are withheld for licence reasons (below),
   so this one can be checked from the report but not recomputed.
5. **H2 (pre-registered): persona customers behave more variedly.** Same
   report, diversity section.
6. **H3 (pre-registered) was not supported.** Plain-LLM vs scripted is
   "different"; persona vs scripted is "inconclusive" by the pre-registered
   rule. We report it that way. Please check we haven't softened it anywhere.
7. **The exploratory family run** (`results/family-sd4/`): other model
   families as customers recover most of the persona arm's variety in opening
   offers but not in wording. Exploratory, no replicates, and Muse made illegal
   moves on 24% of its turns.

## Known weak points (please push on these)

- **The noise floor is a per-episode scale.** It is the mean absolute gap
  between two replays of a cell, compared against a mean over 40 pairs, so it
  does not shrink with n. We kept it because it is the rule we pre-registered,
  and it is conservative, but it decides H3 by 0.001 to 0.002, while the
  floor's own bootstrap interval is about ±0.03. Sonnet's plain-LLM
  "different" also depends on the replicate: on r1 it is +0.064, under the
  floor (0.093). Is it a reasonable bar, and how should we present it?
- **The equivalence margin** (0.08) was half the smallest scripted-vs-LLM gap
  in a pilot (0.16). In the confirmatory run that gap was 0.10 to 0.11, so
  the margin is about three quarters of it. The pilot data is not in this
  repository.
- **Scope.** One small environment, two agents from one model family, Claude
  customers in both LLM arms, 40 seeds, low reasoning effort. Every
  `equiv-sd3` episode ended in a deal, and only 4 Sonnet episodes in
  `family-sd4` did not, so these runs say little about closing rates.
- **Withheld data.** The persona arm's traces carry trait scores from NVIDIA's
  persona dataset; its licence covers use for AI development, but
  redistribution is not clearly covered, so they're not published.
- **The honesty metric is experimental.** One blind labeller (one of the
  authors), two rounds; on the held-out round, bluff precision 64% (95%
  interval 47 to 78%), recall about 38% (23 to 75%), kappa 0.45 (0.28 to
  0.68), reweighted to the held-out pool. Three bugs were fixed after the
  held-out round, and the fixed rules are not re-validated. No finding
  depends on it.
- **Adapters.** The OpenEnv package has been built as a Docker image and
  played end to end through its WebSocket client. The NeMo Gym adapter has only been tested against the
  pinned library, not a live Gym deployment; its training payload (the focal
  seat's token-bearing `response`) is tested through Gym's own rollout-record
  code but not through a NeMo-RL training run.
- **Commit history was re-authored** before publication (authorship only; the
  trees are identical). Commit hashes cited in older notes were updated.

## What an internal review already found

Before asking you, we ran four independent review passes (statistics,
runtime, adapters and security, docs against code and data). What was fixed
is in the pull request that adds this file. Notable fixes:

- the acceptance-by-offer-quality table counted offers the agent wasn't
  authorised to accept as declines, biasing arms unevenly;
- the honesty metric's kappa was quoted on the stratified sample (0.64)
  instead of reweighted to the held-out pool (0.45);
- the OpenEnv server exposed the counterpart's private values and the
  solution through `state()` mid-episode, and could be locked by idle
  sessions;
- task rows sent to the NeMo Gym server could name any importable module as
  an environment;
- OpenAI structured-output refusals were recorded as malformed output and did
  not trip the refusal breaker;
- `verify` passed truncated or edited traces;
- the docs overstated H3 ("partly supported"), described the main metric as a
  share of the deal, and misreported the Muse invalid-move rate and the spend.

Found and deliberately not changed (tell us if you disagree):

- the noise-floor definition (pre-registered; now documented honestly);
- only the first two replicates feed the floor; there's no minimum n before a
  verdict;
- the token cap is checked when an episode starts, so a run can overshoot by
  up to `concurrency` episodes (documented);
- the config hash leaves out the environment and prompt versions, and nothing
  locks a run directory against two processes;
- the Docker image builds on `openenv-base:latest`, installs uv with
  `curl | sh` and runs as root; CI pins actions by tag, not SHA.

## Questions we'd most like answered

1. Does any conclusion in `results/README.md` or the README go beyond what
   the reports show?
2. Is the matched-pair design, with the replay noise floor and equivalence
   margins, sound for this kind of comparison? What would you change?
3. Are the support-desk metrics measuring what they say (see the metric
   definitions in `docs/concepts.md`)?
4. Could you plug in an environment of your own from
   `docs/custom-environments.md` alone?
5. Anything in the repository that shouldn't be public?

## How to send feedback

Comment on the lines in [#2](https://github.com/Parley-Labs/parley-sim/pull/2),
or open an issue for anything broader. For each point, the file and
line, what you saw, and how to reproduce it is ideal, but a rough note is
welcome too.
