# AD-D2: a world-independent advantage scale (development)

Registered 2026-11-23 16:17 IST.
- **What had been seen:** the AD-D1 seed-0 outputs of all four arms and the three baselines. Seed 1 had not been
  read; tasks 1205 and 1206 were running.
- **Hash file:** `results/ad_hashes_d2.json` (`scripts/ad/freeze_d2.py`).
- **D1 untouched:** no D1-hashed file is edited, and D1 tasks 1207–1211 run unchanged.

## 1. Diagnosis that motivates D2 (seed 0, development, descriptive)

These are the training rollouts at T = 1, in 20-step blocks (`scripts/ad/analyze_d1.py`).

| Arm | Reads per episode | Verify after pay | Zero-std groups |
|---|---|---|---|
| grpo | 0.59 → 0.03 | 0.21 → 0.01 | 19% → 86% |
| nostd | 0.69 → 1.56 | 0.24 → 0.52 | — |
| awrbn | 0.77 → 1.35 | 0.27 → 0.46 | — |
| awrb | 0.94 → 2.25 | 0.32 → 0.77 | — |

- **grpo collapses verification,** as predicted by P1. The registered falsifier did not occur.
- **The no-std arms move in the VoI direction, but slowly.** At the same learning rate their effective step is about
  1/σ smaller than GRPO's: σ ≈ 0.13–0.16 is the mean within-group std.
- So "nostd vs grpo" at one learning rate confounds the advantage's *direction* with its *step size*.

## 2. Change (one)

Keep the group-mean centring and drop the per-group std. Divide every advantage at step t by one scalar:

  s_t = max(0.02, EMA_{0.9} of the mean within-group std over steps < t)

The divisor at step 0 uses its own batch. This is `actionproof/ad/scale.py`, which wraps TRL's
`_generate_and_score_completions`.

The divisor is independent of the current step's hidden worlds, so the expected update keeps the VoI sign (P2, P5).
The step size matches GRPO's typical scale.

Everything else is as in AD-D1:
- cw_train;
- c = 0.05;
- warm start, 200 steps, 4 × 8 rollouts;
- lr, β, LoRA as before.

## 3. Arms and runs

| Arm | Return | Scale |
|---|---|---|
| `nostd_ema` | R_V − c·reads | EMA |
| `awrbn_ema` | E[R_V \| traj] − c·reads | EMA |

Seeds 0 and 1. Tasks:
- 1212 nostd_ema s0;
- 1213 awrbn_ema s0;
- 1214 awrbn_ema s1;
- 1215 nostd_ema s1.

They run after the D1 chains, one chain each, so there are still at most two GPUs. Task 1212 begins with a 2-step
smoke check that the scale log is written.

## 4. Gates (cw_dev, greedy, mean of seeds 0–1)

Applied to each arm, against D1's grpo seeds 0–1. These are the D1 gates G1–G4 unchanged:

| Gate | Requirement |
|---|---|
| G1 | VSC ≥ grpo + 0.03, and each seed positive |
| G2 | J_c ≥ grpo |
| G3 | Duplicates ≤ grpo + 0.02 |
| G4 | KL₁₀₀ ≤ 10× grpo's |

**Additional development comparisons** (descriptive, two seeds):
- **AWRB at matched scale:** awrbn_ema − nostd_ema, on VSC, J_c and verify-after-pay.
- **Scale effect:** nostd_ema − nostd (D1), and awrbn_ema − awrbn (D1).

## 5. Registered predictions

- **Q1.** nostd_ema and awrbn_ema reach a higher verify-after-pay rate at step 100 than nostd and awrbn did.
- **Q2.** Neither collapses verification the way grpo did: no 20-step block after step 40 has a read rate below its
  first block's.
- **Q3 (no prediction of sign).** awrbn_ema vs nostd_ema. AD-T0b predicts an advantage for AWRB only if repair
  succeeds often. A tie is a negative result for AWRB's added value.
