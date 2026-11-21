# S3 addendum 1: correction and a class-matched start-state control

Registered 2026-11-21, before any S3 confirmatory output was read. The S3 evaluation jobs were running at that time, but none of their outputs had been fetched or opened. The registration time is the commit time of this file and `results/s3a1_hashes.json`. `PREREGISTRATION_S3.md` is unchanged, and its H1 and H2 are tested exactly as registered.

## 1. Correction to S3 §2

S3 §2 says the only change from A12 R_V is the start-state distribution. **That is not correct.** RS-GRPO's pool is 200 `proc_train_b` tasks plus 200 recovery starts. The recovery starts contain fault types that `proc_train_b` does not:
- duplicate delivery;
- partial ratios ¼ and ¾;
- complete and fully done states.

A12 R_V also saw 400 distinct full-episode tasks, against RS-GRPO's 200. So the S3 contrast (H1, H2) measures **the RS-GRPO training recipe against A12 R_V** (start states plus fault coverage). It does not isolate the start state. This was raised by an independent review (`docs/stage3/ACADEMIC_REVIEW.md`).

## 2. Class-matched control

**Baseline `s3_matched`.** R_V, with the same trainer, settings, warm start, steps and seeds as RS-GRPO. The pool:
- the same 200 `s3_trainb_half` tasks;
- 200 **full episodes** built from the 200 recovery-start tasks themselves (`scripts/s3_make_matched.py`).

Each full episode keeps its source task's world, prompt and fault: same kind, same step, same partial ratio, and never-commit on the same tool. "Complete" and "done" starts become clean episodes. Only the replayed prefix is removed.

**What differs.** The two pools differ only in where an episode starts.

**Runs.** Seeds 0–2 (`runs/s3_matched_s<seed>`; tasks 915–917), evaluated once on every S3 confirmatory split (task 924).

## 3. Hypotheses (added; S3 H1 and H2 unchanged)

Comparison: RS-GRPO seed s minus `s3_matched` seed s, on the same tasks, seeds 0–2. Statistics: `scripts/s3_stats_addendum.py` (the frozen `s3_stats.py` with the baseline and an optional class filter).

- **H3 (start-state effect, recovery starts):** VSC on `s3_conf_rec` is higher for RS-GRPO. Supported if positive in 3 of 3 seeds **and** the seed-averaged 95% interval excludes 0. With 3 seeds, the exact sign-flip resolution is 1/8 one-sided.
- **H4 (start-state effect, full episodes, the classes recovery starts target):** VSC on `s3_conf_full` restricted to the partial and duplicate-delivery classes is higher for RS-GRPO. Same rule.
- **Wording rule.** RS-GRPO is described as a **start-state curriculum** applied to GRPO, not a new optimisation algorithm. A benefit *of starting from recovery states* is claimed only if H3 or H4 holds. If neither holds, the S3 result (if positive) is attributed to the recipe as a whole, and coverage of fault types is named as a sufficient explanation.
- **Secondary, pre-specified:**
  - per-class and per-workflow (`order_prepay`, `receive_pay`, `pay_post`) breakdowns of H1–H4;
  - S3 H1 and H2 computed separately on seeds 0–1 (trained before the S3 registration) and seeds 2–4 (trained after it).

## 4. Compute

- Training: tasks 915–916 on the Slurm H200 lease, task 917 on HF.
- Evaluation: task 924, on Slurm after training.
- Missing runs are reported as missing.
