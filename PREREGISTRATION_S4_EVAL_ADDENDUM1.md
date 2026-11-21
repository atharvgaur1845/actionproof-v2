# S4-EVAL addendum 1: an attainable multiplicity rule

Registered 2026-11-21, about 20:00 IST, before any S4-EVAL output exists. The evaluation jobs (tasks 950–953) are queued on the Slurm H200 lease behind the FCA development runs; `results/s4/eval/` does not exist on the cluster at the time of this commit. The registration time is the commit time of this file and `results/s4eval_a1_hashes.json`. `PREREGISTRATION_S4_EVAL.md`, its splits and `results/s4eval_hashes.json` are unchanged. The splits, models, evaluation loop and hypotheses G1–G4 are unchanged.

## 1. The error

`PREREGISTRATION_S4_EVAL.md` §3 says "With five seeds the smallest one-sided p is 1/32, so G1–G4 can all survive only if each is 5 of 5." That is wrong. Holm's first step over four tests compares the smallest p with 0.05 / 4 = 0.0125. The smallest exact sign-flip p at five seeds is 1/32 = 0.031, so the smallest Holm-adjusted p is 4/32 = 0.125. **No hypothesis could be supported at 0.05, whatever the data.** This was found by the post-results review (`docs/stage3/ACADEMIC_REVIEW_POST.md`, §1.3 and §4 item 5).

## 2. The amended decision rule

The seed-level statistic changes from the paired sign-flip to an **unpaired seed permutation**:
- Pairing RSC seed s with A12 R_V seed s shares no randomness. The two arms were trained on different pools with different code paths, so seed index s means nothing common to both.
- The test enumerates all C(10, 5) = 252 assignments of the ten per-seed VSC values to two groups of five. The statistic is the difference of group means, one-sided (RSC higher).
- The smallest p is 1/252 = 0.004, so Holm over four tests is attainable (4/252 = 0.016).

A hypothesis Gk is **supported** if, on each of its splits:
1. at least 4 of 5 seed-index differences are positive (as registered);
2. the seed-averaged 95% task-bootstrap interval excludes 0 (as registered);
3. **and** the Holm-adjusted unpaired seed-permutation p is below 0.05, with Holm across G1–G4.

For G2, both `s4_lang_hi` and `s4_lang_hinglish` must meet (1) and (2), and its p is the larger of the two splits' p-values (an intersection-union test).

The exact sign-flip p is still reported for every split, unadjusted and descriptive.

## 3. Added descriptive outputs (no decision rule)

- A two-level bootstrap interval (resample seeds within each arm, then tasks), 2,000 draws.
- The fraction of tasks on which two seeds of the same arm make identical tool calls (seed redundancy).
- Each fault class's contribution to the RSC − A12 R_V VSC difference, in units of the whole split.
- For the start-state comparison (RSC − class-matched control, seeds 0–2), the unpaired permutation over 3 vs 3 seeds (smallest p 1/20) and over RSC 5 vs control 3 (smallest p 1/56). Any later control seeds are added only as labelled extra rows; the 3-seed result is reported first.

## 4. Script

`scripts/s4_stats.py` implements §2 and §3, and the `s3_stats.py` paired bootstrap and sign-flip. Before this commit it was run only on S3 confirmatory data that is already published (`--dir results/s3/conf --splits s3_conf_rec,s3_conf_full`), to check that it reproduces the review's seed-level numbers. It reads `results/s4/eval/` and writes `results/s4/s4_summary.json`.
