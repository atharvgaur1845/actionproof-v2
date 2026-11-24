# Judge calibration results (model vs verifier; no human labels)

Source: `results/j2/j2_summary.json` via `docs/judges/_j2_tables.md`. Evidence: exploratory.

## Separation of fitting and evaluation
- The aggregation weights (per-judge P(answer | goal_met), add-one smoothing, class prior) were fit on the **76
  calibration cases** and evaluated on the **660 main cases**. The splits are disjoint
  (`data/cases.jsonl` split field; PREREGISTRATION_J1 §5).
- Individual judges' Brier score and ECE (5 bins) use their own stated confidences on the main set. No refitting.

## Calibration of stated confidence (Q1, main set)

| View | Judge | Brier | ECE |
|---|---|---|---|
| transcript_state | gemma | 0.011 | 0.011 |
| transcript_state | gptoss | 0.051 | 0.047 |
| transcript_state | qwen | 0.102 | 0.080 |
| transcript_state | deepseek | 0.105 | 0.106 |
| transcript_state | aya | 0.297 | 0.276 |
| transcript | gemma | 0.106 | 0.030 |
| transcript | aya | 0.338 | 0.317 |

The full table, including qwen, gptoss and deepseek on transcript only, is in `_j2_tables.md`.

## Aggregators (main set)

| View | Weighted (four) F1 / Brier / ECE | Majority (four) F1 / Brier / ECE | Best single (gemma) F1 |
|---|---|---|---|
| transcript_state | 0.963 / 0.026 / 0.033 | 0.896 / 0.044 / 0.069 | 0.990 |
| transcript | 0.828 / 0.131 / 0.125 | 0.635 / 0.236 / 0.295 | 0.800 (coverage 0.576) |

## By failure category
The fooled-on-unverifiable-failures stratum (120 transcript-only cases): Aya 0.875, the other judges 0–0.008. Other
per-category breakdowns are in the J1 report (`results/j1/j1_summary.json`); J2 adds no new category analysis.

## Decision for the paper
Report the verifier as the execution-truth instrument. Report judges only as a measured limitation of
transcript-based evaluation:
- the policy-family judge accepts 70% of failed executions;
- the other judges trade false acceptances for abstention.

Do not report a calibrated ensemble as an evaluator. It does not beat the best single judge with the snapshot, and it
has no human validation.
