# FCA development result: dropped (2026-11-21)

**Verdict.** FCA (forked counterfactual action credit, `FCA_GRPO.md`) fails the first kill criterion, which was fixed before training. It is dropped. No confirmatory comparison is registered, and FCA is not trained on seeds 2–4.

Numbers: `results/s4/fca_dev_summary.json` (`scripts/s4_fca_dev.py`). Inputs:
- dev summaries in `results/s4/dev/` and `results/s3/dev/`;
- trainer states and fork-group logs in `results/s4/train_state/`.

## Criteria (development seeds 0–1, against recovery-start GRPO with the same pool, steps, rollouts and seeds)

| Criterion | Threshold | Result | Kill? |
|---|---|---|---|
| `s3_rec_dev` VSC, FCA − RSC, mean of seeds 0–1 | at least +0.03 | **−0.097** (seed 0: 0.640 vs 0.807; seed 1: 0.780 vs 0.807) | **yes** |
| `proc_dev` VSC, FCA − RSC | no fall greater than 0.03 | −0.017 (0.827 vs 0.860; 0.860 vs 0.860) | no |
| KL at step 100, FCA / RSC | at most 10× | 1.08× at most | no |

## Where it lost (`s3_rec_dev`, VSC per class, seeds 0 and 1)

| Class (n) | FCA | RSC |
|---|---|---|
| partial, acknowledgement start (16) | 0.19, 0.81 | 0.81, 0.75 |
| partial, read start (18) | 0.17, 0.67 | 0.67, 0.78 |
| duplicate, acknowledgement start (9) | 0.44, 0.67 | 0.89, 0.78 |
| duplicate, read start (14) | 0.64, 0.64 | 0.79, 0.79 |
| every other class | identical | identical |

FCA seed 0 did not learn partial-payment repair, like class-matched control seeds 1 and 2 (Stage 3 addendum 1). False completion on `s3_rec_dev` rose to 0.25 in seed 0, against 0.09 for RSC.

## Diagnostics (training logs; descriptive)

- 382 fork groups per run.
- On-policy rollouts in a fork group had zero reward variance in 60% (seed 0) and 50% (seed 1) of groups.
- In 66% and 72% of fork groups, at least one forced action class reached a mean reward different from the on-policy mean. In those groups FCA's forced branches did carry a learning signal that plain GRPO would not have had.
- That extra signal did not turn into better recovery. Two explanations fit the data, and neither was tested:
  - Fork groups give half their rollouts to forced branches (4 of 8 on-policy). This halves the on-policy samples from which a repair is discovered.
  - The credit goes to the class of the first action only. When the continuation after a forced repair fails, the repair class gets negative credit.

## What this means for the paper

- FCA is reported as an honest negative result for a known estimator family (all-action and counterfactual-baseline credit: COMA, BiPACE, GiGPO, CRR) in this setting.
- It is not a contribution claim. Its positive analogue was never claimed (`DECISION.md`).
- Two seeds on development data cannot show that FCA is worse in general. They show only that it did not meet the bar set before training.
