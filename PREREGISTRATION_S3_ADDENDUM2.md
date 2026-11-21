# S3 addendum 2: five control seeds, and is the receive-then-pay failure a pool-frequency effect?

Registered 2026-11-21, before any model named here is trained and before any model is evaluated on the new splits. The registration time is the commit time of this file and `results/s3a2_hashes.json`. `PREREGISTRATION_S3.md`, `PREREGISTRATION_S3_ADDENDUM1.md` and their results are unchanged and stay the registered S3 results.

Both parts respond to the post-results review (`docs/stage3/ACADEMIC_REVIEW_POST.md`, §2.4, §2.8 and §4 items 1 and 3). **Disclosure:** the seed 0–2 results of RSC, the class-matched control and A12 R_V on the S3 confirmatory splits are known. Part A adds two new control seeds to a comparison whose first three seeds were already seen, so the 5-seed test is not independent of them. The seeds 3–4-only contrast is reported next to it.

## Part A: the start-state comparison with five control seeds

**Runs.** `s3_matched` seeds 3 and 4: the same pool, trainer, settings and warm start as seeds 0–2 (tasks 960–961). Each is evaluated once on `s3_conf_rec`, `s3_conf_full`, `s3_conf_stress` and both OpsSim splits (task 966), with the frozen S3 evaluation.

**Hypotheses.** The contrast is RSC (`s3_rv_mix`, seeds 0–4) minus the class-matched control (seeds 0–4), on VSC.

| | Tasks | Prediction from seeds 0–2 |
|---|---|---|
| H3′ | `s3_conf_rec`, all 300 | positive |
| H4′p | `s3_conf_full`, partial class (64) | positive |
| H4′d | `s3_conf_full`, duplicate-delivery class (46) | not positive (the control was equal or better) |

- **Statistic:** the one-sided unpaired seed permutation over all 252 assignments of the ten per-seed VSC values (RSC higher), with Holm across the three.
- **Supported if** the Holm-adjusted p is below 0.05 **and** the seed-averaged 95% task-bootstrap interval (RSC mean minus control mean, tasks resampled) excludes 0.
- **Wording rule.** A start-state benefit is claimed only for the classes whose hypothesis is supported. If H4′d is not supported, no start-state benefit is claimed for duplicates.
- **Reported next to it (descriptive):**
  - the registered 3-seed results (addendum 1);
  - the seeds 3–4-only contrast;
  - a two-level bootstrap (seeds, then tasks);
  - stress, and duplicates left unrefunded;
  - the number of control runs that learn partial-payment repair (partial-class VSC ≥ 0.40 on `s3_conf_full`), against RSC's 5 of 5.

## Part B: workflow-rebalanced pools

**Question.** Partial payments after a goods receipt (`receive_pay`) are repaired by no arm: RSC repairs 5 of 60 full episodes. The RSC pool has 6 such tasks out of 400, against 32 order-then-pay and 29 pay-first. Is the failure a pool-frequency effect, or a property of the evidence context?

**Pools** (`scripts/s3_make_rebal.py`):
- Pool size, every fault-class count, every partial ratio and every start type are kept.
- Randomly chosen order-then-pay and pay-first partial-payment tasks are replaced, in place, by fresh receive-then-pay tasks with the same fault dict and start type.
- Result: partial-payment tasks per workflow go from 32 / 29 / 6 to 23 / 23 / 21 (order then pay / pay first / receive then pay).
- Files: `s3_trainb_half_rb.jsonl` + `s3_rec_train_half_rb.jsonl` (recovery-start recipe), and `s3_trainb_half_rb.jsonl` + `s3_matched_full_half_rb.jsonl` (class-matched control, derived as in addendum 1).

**Runs.** Rebalanced RSC `s3_rb_mix` seeds 0–1 and rebalanced control `s3_rb_matched` seeds 0–1. Same trainer, settings and warm start as RSC (tasks 962–965).

**Targeted splits** (fresh worlds; ids disjoint from every split):
- `s3_pp_full`: 150 full episodes with a partial payment on the first `pay_supplier` attempt. 50 per workflow; ratios ¼, ½ and ¾ in turn.
- `s3_pp_rec`: 150 recovery starts placed right after the faulted payment, alternating acknowledgement and read starts, with the same balance.
- Scripted calibration: the oracle scores 1.00 and never-read 0.00 in every workflow on both splits (`results/s3/scripted_pp_calibration.json`).

**Evaluation.** Each of these models is evaluated once on both targeted splits (tasks 967–968): the warm start, A12 R_V 0–4, RSC 0–4, the control 0–4, rebalanced RSC 0–1 and rebalanced control 0–1. The four rebalanced models are also evaluated once on every S3 confirmatory split (task 969; secondary).

**Primary quantity (W).** Receive-then-pay VSC of rebalanced RSC, averaged over its two seeds and over the 100 receive-then-pay tasks of `s3_pp_full` and `s3_pp_rec`. The comparison is with RSC seeds 0–1 on the same tasks (seed-averaged task bootstrap). Interpretation, fixed now:

| Result | Reading |
|---|---|
| W ≥ 0.50 and the interval vs RSC excludes 0 | Pool frequency explains most of the receive-then-pay failure |
| W ≤ 0.15 | Not explained by pool frequency at 3.5 times the original count; the evidence context (a payment read after a goods receipt) remains the explanation to test |
| otherwise | Partly frequency; inconclusive |

With two seeds this is a two-run result, and it is reported as such.

**Secondary (descriptive):**
- the same quantity for the rebalanced control;
- order-then-pay and pay-first cells of every model on the targeted splits, to check that the lower counts there (32 → 23, 29 → 23) did not cost repair;
- every original model's per-workflow VSC on the targeted splits, which replaces the 12-task cell of `s3_conf_full` with 50 tasks per workflow;
- the rebalanced models on `s3_conf_rec` and `s3_conf_full`.

## Part C: the zero-variance mechanism (no GPU)

The review notes that "GRPO rarely samples the repair from a full episode" was argued, not measured. From the existing rollout logs (`results/s3/rollouts/*.jsonl.gz`), compute for each run the share of partial-fault groups whose rewards have zero variance, by workflow and training step, for RSC seeds 0–4 and the control seeds.

**Not pre-registered.** `scripts/s3_zero_var.py` was run on the seed 0–4 RSC and seed 0–2 control logs while it was being checked, before this commit, and its output was seen. Part C is therefore a descriptive analysis with no prediction and no decision rule. Control seeds 3–4 are added when their logs exist.

## Statistics and scripts

- `scripts/s3_stats_a2.py` (Parts A and B); `scripts/s3_zero_var.py` (Part C).
- Both use the S4-EVAL addendum 1 helpers (`scripts/s4_stats.py`: unpaired permutation, Holm, two-level bootstrap).
- Missing runs are reported as missing.
