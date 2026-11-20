# Amendment 12, addendum 1: rebalanced training pool and gate at the training group size

Registered 2026-11-20 (commit time of this file and `results/a12b_hashes.json`), after the A12 gate and **before any GRPO run**. Decision by the user, 2026-11-20 03:00 IST.

## What happened

The registered gate (`results/a12/gate.json`, sampled T = 1 x 4 on 100 `proc_train` tasks, warm start `runs/rft_proc/final`) passed three checks and failed one:

| check | value | band | result |
|---|---|---|---|
| phantom read-back | 0.309 (58/188) | [0.10, 0.60] | pass |
| verified completion | 0.600 | [0.25, 0.75] | pass |
| tool-call validity | 0.989 | >= 0.95 | pass |
| tasks with reward variation (R_V, R_A) | 0.15, 0.17 | >= 0.30 | **fail** |

The warm start is not deterministic: in 75 of the 100 tasks its 4 samples differ in whether they read back. The low share is structural: the reward-truth table shows that reading cannot change either reward in the clean and precommit-timeout classes (48% of `proc_train`), and changes R_V only in phantom and partial tasks and R_A only in lost-ack and never-commit tasks. The registered re-mix rule (change the demonstrator share) addresses read-back out of band, which is not the failure here, so it is not applied.

## Changes (nothing else changes)

1. **Training split** `proc_train_b` (`scripts/make_proc_train_b.py`, seed 109, 400 tasks, PO ids disjoint from every split): same generator and classes, rebalanced mix: clean 0.05, phantom 0.35, partial 0.15, lost_ack 0.25, precommit_timeout 0.05, never_commit 0.15. Realized: phantom 139, lost_ack 110, never_commit 67, partial 52, clean 18, precommit_timeout 14. The scripted oracle solves every attainable task (checked).
2. **Gate at the training group size.** The informative-group check is measured with 8 samples per task (the GRPO group size; the 4-sample value was described in A12 §4 as a conservative stand-in), on the first 100 `proc_train_b` tasks, threshold unchanged (>= 0.30 for R_V and for R_A). The other three checks are re-measured on the same samples with unchanged bands. Warm start unchanged (`runs/rft_proc/final`, sha256 in the run inputs). Output `results/a12b/gate.json`; the H3 step-0 predictions are recomputed on these samples (`results/a12b/h3_predictions.json`) and committed before GRPO step 1. The A12 predictions file (`results/a12/h3_predictions.json`) is kept and reported too.
3. If this gate fails, A12 ends as registered ("no learnable band"); there is no further re-mix.

Unchanged: rewards, reward-truth table, training settings, seeds and their rule, checkpoint protocol, confirmatory splits (`proc_conf`, `proc_combo`, `proc_stress`, still drawn from the original mix), endpoints, statistics, claim criteria.

Effect on interpretation: training sees more of the classes where the two rewards disagree, so effects may appear in fewer steps; the confirmatory test is still on the original mix.
