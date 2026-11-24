# AD-C1: registered outcome vs the theory's frozen predictions

**Status (2026-11-24): pending.**
- Training: 1222 (grpo s4) and 1224 (awrb s3) were running at the last check. Evaluation tasks 1240/1241 follow.
- The registered analysis (`scripts/ad/c1_stats.py`) is run by the lead engineer, once.
- This branch has **not** opened any cw_conf output.

## Frozen predictions
`docs/research/AD_C1_THEORY_PREDICTION.md` (commit 2354929; hash in `results/ad_c1_theory_prediction_hash.json`).

| ID | Prediction |
|---|---|
| T1 | H_nostd supported; seed-averaged Δ VSC ∈ [+0.05, +0.20] |
| T2 | grpo reads per episode ≤ 0.2 in every seed; nostd ≥ 0.8 in ≥ 4/5 |
| T3 | ≥ 70% of the gain is on faulted tasks; clean-task \|Δ\| ≤ 0.03 |
| T4 | false-success gap ≈ VSC gap (± 0.05) |
| T5 | H_AWRB not supported |
| T6 | awrb reads more than nostd in ≥ 4/5 seed pairs |
| T7 | grpo seed-level VSC range ≤ 0.02 |
| T8 | untestable |

## How the comparison will be filled (after c1_stats exists)

For each of T1–T7 (T8 is untestable), record:
- observed value;
- source file and key;
- met / not met.

Per-seed values are listed individually. Effect sizes and intervals are the registered ones, copied, not recomputed.
A failed prediction is reported as a failure of the theory's link to the language agent, not of the mathematics.
