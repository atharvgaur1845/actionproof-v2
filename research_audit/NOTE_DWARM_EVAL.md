# Descriptive note: evaluating P1's recovery-SFT warm start (`p1_dwarm`)

Written 2026-11-25, 09:15 IST, **before any output** of this evaluation exists. It is not a registration, and no hypothesis test is attached to it. Every number it produces is descriptive.

**Why.** `P1_DECOMPOSITION.md` found that P1's arms C and D are not nested. C is A plus recovery SFT. D is the RFT warm start plus recovery SFT (`p1_dwarm`) plus GRPO. D's starting point `p1_dwarm` was never evaluated, so the gap between 0.772 and 0.819 does not measure what GRPO adds. This evaluation fills the missing cell of the 2×2:

| | no GRPO | GRPO (RS-GRPO recipe) |
|---|---|---|
| RFT warm start | W (`rft_proc`) | A (`s3_rv_mix_s*`) |
| + recovery SFT | **Dw (`p1_dwarm`, this note)** | D (`p1_d_s*`) |

**What runs.** Task `slurm/tasks/1620_p1_dwarm_eval.sh`. Model `runs/p1_dwarm/merged`, no adapter, tag `Dw`. Splits:
- dev: `p1_dev_full`, `p1_dev_rec`;
- frozen test: `p1_test_{full,pp,rec,wf4,combo}`.

Settings are the same as every P1 arm: `scripts/p1_eval.py`, 4 samples, the unchanged verifier. It is a single checkpoint (the SFT has one seed), so only task-level uncertainty is reported.

**How it is read** (fixed now):
- **GRPO increment on top of recovery SFT:** D (3-seed mean) − Dw.
- **Increment on the plain warm start:** A − W.
- **Interaction:** the difference between those two increments.
- Paired task bootstrap over the 450 test tasks with 10,000 resamples. Each contrast is reported with its 95% interval and labelled exploratory.
- Nothing is called "supported", and no registered P1 or P3 result is changed.
- If the P3 result is available, Dn − Dw is reported the same way.
