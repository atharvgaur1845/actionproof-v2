# PREREGISTRATION AD-C2: does GRPO's learned verification respond to the cost of verifying? (limited intervention)

Registered: written 2026-11-28 17:50 IST; the registration is the commit that adds `results/ad_c2_hashes.json`
(`scripts/ad/freeze_c2.py`), which hashes this file, the training and evaluation code, the task scripts and the sealed
split. No model has been run on `cw_c2` at registration; `freeze_c2.py` refuses to register otherwise.

## 1. Question and why it is limited

AD-C1 left one frozen prediction untestable (T8): no plain-GRPO language-model run at a second verification cost.
The theory (Theorem 3.1 and Corollary 3.2 in `docs/theory/PROOF_AND_ASSUMPTIONS.md`) says that, when all rollouts of a
group share one hidden fault plan, GRPO's expected update follows the vote V = Σ_z w(z) sgn(β_z), so it does not
depend on the read cost c as long as c is large against ε/σ̂ (c ≳ 10⁻³ at ε = 10⁻⁴) and no state's sign changes. A
centred (std-free) estimator follows the magnitude-weighted gap ΔQ, which grows as c falls. At c = 0, clean groups that
differ only in reads have zero reward spread and contribute no update, so the vote is carried by faulted states alone.

This is a **limited intervention**: one training seed per new arm, one group law (shared, as in AD-C1), and cost
values below the AD-C1 value only. The planned full factorial (2 estimators × 3 costs × 2 group laws × seeds) does not
fit the remaining time; the iid group law needs new rollout code that cannot be built and checked in time. Neither is
claimed.

## 2. Design

- **Sealed split.** `data/bench/proc/cw_c2.jsonl`, 300 tasks (120 faulted, 180 clean), seed 1204, the CW world prior
  `cw1` of AD-C1; written 2026-11-28 17:48 IST by `scripts/ad/c2_split.py`, which asserts instance-disjointness (task
  ids, PO ids, prompts) from `cw_train`, `cw_dev` and `cw_conf`. sha256 `0f792406c9ed3856ca01293479c4aadfe3602f6a5934658e69ddad330a5867cb`.
- **New arms (training only, `slurm/ad_arm_c2.sh`).** The AD-C1 recipe unchanged — Tiny Aya Global 3.35B, LoRA,
  warm start `runs/rft_proc/final`, `cw_train`, 200 steps, the same estimators' code and group construction (one fault
  plan per group) — except `--cost`:
  - `grpo_c001`: GRPO (group std), c = 0.01, seed 0;
  - `nostd_c001`: std-free centred estimator, c = 0.01, seed 0;
  - `grpo_c000`: GRPO, c = 0 (boundary arm), seed 0.
- **Reference arms (existing, c = 0.05).** AD-C1's `ad_grpo_s0` and `ad_nostd_s0`, re-evaluated on `cw_c2`.
- **Evaluation.** Each of the five models once on `cw_c2`: greedy + 4 sampled episodes at T = 1
  (`actionproof.ad.evaluate`, unchanged). Task `2003_c2_eval_stats` runs all evaluations and then the analysis once.

## 3. Predictions (derived from the theory before any output)

| ID | Prediction | Basis |
|---|---|---|
| P1 | `grpo_c001` is cost-blind: greedy reads per episode ≤ 0.2 on cw_c2 (the no-read corner of GRPO at c = 0.05) | Cor. 3.2: the vote has the same sign at c = 0.01 and 0.05 |
| P2 | `nostd_c001` reads: greedy reads per episode ≥ 0.8 | centred update follows ΔQ, larger at lower c |
| P3 | At c = 0.01, std-free beats GRPO on VSC: lower 95% bound of D1 = VSC(nostd_c001) − VSC(grpo_c001) > 0 | as AD-C1 T1, at a second cost |
| P4 (secondary) | `grpo_c000` reads: greedy reads per episode > 0.2 | at c = 0 the clean-state vote vanishes; faulted states vote to read |
| P5 (secondary) | std-free reads at least as much at c = 0.01 as at 0.05: D3 estimate ≥ 0 | ΔQ decreases in c |

Falsification: P1 failing (GRPO reads at c = 0.01) contradicts the claim that the language agent's GRPO update is
cost-blind in this regime. P4 failing means the c = 0 discontinuity does not appear in the language model; it would
be reported as a failure of the boundary prediction, not explained away.

## 4. Analysis (`scripts/ad/c2_stats.py`, hashed)

A model's task score is the mean over its 5 episodes. Paired differences over the 300 tasks, 95% percentile intervals
from 10,000 task resamples (seed 20261128): D1 (above), D2 reads(grpo_c001) − reads(grpo_c005), D3
reads(nostd_c001) − reads(nostd_c005), D4 reads(grpo_c000) − reads(grpo_c001), D5 VSC(grpo_c000) − VSC(grpo_c001).
Reported per arm: greedy and all-5 reads, VSC, false success, faulted-task VSC, verify-after rates, and net utility
VSC − c·reads at the arm's training cost. One seed per arm: intervals describe evaluation uncertainty with each model
fixed; no seed-level claim and no multiplicity correction is made. Predictions are checked exactly as written above.

## 5. Exclusions and deviations

No exclusions. An infrastructure failure before an output file exists means the same command is rerun; a partial file
is never used. A run that does not finish before the portal cutoff is reported as not run; no result is imputed.
Deviations go in a dated addendum; this file is not edited after the first cw_c2 output.
