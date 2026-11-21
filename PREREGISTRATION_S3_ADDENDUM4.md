# S3 addendum 4: is the recovery-start effect the evidence handed over in "read" starts?

Registered 2026-11-22, before any model named here is trained and before the outputs of S3 addendum 2 (control seeds 3–4, the rebalanced pools) are read. The registration time is the commit time of this file and `results/s3a4_hashes.json`. Addendum 3 was withdrawn before any training (`docs/research/ADDENDUM3_WITHDRAWN.md`). Every other registration and result is unchanged.

## 1. Question

In the RSC pool, 97 of 200 recovery starts are "read" starts. Their scripted prefix ends with the authoritative read after the faulted write, so the conversation hands the agent the record (for example, a payment of a quarter of the amount) just before its decision. The registered start-state effect is on partial payments only. One explanation is that read starts give the model in-context examples of evidence followed by a decision, which full episodes almost never sample (`docs/RESEARCH_RED_TEAM_REVIEW.md`).

## 2. Arms

| Arm | Pool | Runs |
|---|---|---|
| CONTROL (`s3_matched`) | full episodes with the recovery starts' faults | seeds 0–4 (addendum 1 and addendum 2) |
| ACK (new, `s3_ackonly`) | the RSC pool, with every read start converted to the acknowledgement start of the same task (`scripts/s3_make_ackonly.py`; 97 converted, 9 "done" starts unchanged) | seeds 0–4 |
| RSC (`s3_rv_mix`) | the RSC pool | seeds 0–4 (S3) |

- ACK uses the unchanged Stage 3 trainer and rollout (`slurm/s3_arm.sh`), the R_V reward, the same warm start, settings and steps, on the Slurm H200 lease.
- It differs from RSC only in the dropped read call at the end of 97 prefixes.

## 3. Hypotheses

The metric is partial-class VSC per training run on `s3_conf_full` (64 partial tasks, full episodes, identical for every arm).

| | Contrast | Supported if |
|---|---|---|
| D1 | RSC − ACK (handed evidence) | Holm-adjusted one-sided unpaired seed permutation p < 0.05 (252 assignments; Holm over D1 and D2), **and** the seed-averaged 95% task-bootstrap interval excludes 0 |
| D2 | ACK − CONTROL (starting at the decision without handed evidence) | same |

| Result | Reading |
|---|---|
| D1 only | The handed evidence in read starts carries the effect |
| D2 only | Starting at the decision point carries it; the handed read adds nothing measurable |
| both | Both contribute |
| neither | Not separable with five runs per arm (reported as inconclusive) |

**Sensitivity.** With five runs per arm, only large differences in the number of runs that learn partial repair are detectable (5 of 5 against 1 of 5 learners gives p ≈ 0.02). This is stated in advance.

**Secondary (descriptive):**
- the same contrasts on `s3_pp_full` and `s3_pp_rec` (addendum 2's targeted splits; partial payments only, 50 per workflow);
- the duplicate class;
- `s3_conf_rec`;
- runs that learn partial repair (partial-class VSC ≥ 0.40);
- a template-cluster bootstrap (resampling the task generator's templates).

## 4. Compute and scripts

- Tasks 991–995: training. Task 996: S3 confirmatory splits. Task 997: the targeted splits. All run on the Slurm H200 lease.
- Statistics: `scripts/s3_stats_a4.py`.
- Missing runs are reported as missing.
