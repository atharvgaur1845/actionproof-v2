Groups: 100 tasks x 4 samples (warm start, T=1).

Informative groups (non-zero reward variance): M1 pcstr_d 32, M2b noND (F_under kept) 32, M2 noreport 23, M3' outcome_G 18, M3 outcome 18

| behaviour | n | M1 pcstr_d | M2b noND (F_under kept) | M2 noreport | M3' outcome_G | M3 outcome |
|---|---|---|---|---|---|---|
| completed on RP (G=1) | 93 | +23.04 | +23.04 | +17.84 | +17.86 | +17.86 |
| completed on OK (G=1) | 83 | +1.73 | +1.73 | +1.73 | +1.73 | +1.73 |
| completed on RP (G=0) | 82 | -10.85 | -10.79 | -10.79 | -12.81 | -12.81 |
| escalated on F5 | 68 | +22.58 | +22.60 | +11.55 | +11.55 | +11.55 |
| completed on UR (G=0) | 38 | -5.46 | +5.46 | +5.46 | +0.00 | +0.00 |
| failed on UR | 10 | +5.46 | -5.46 | -5.46 | +0.00 | +0.00 |
| completed on F5 (G=1) | 9 | -10.46 | -10.48 | +0.58 | +0.58 | +0.58 |
| completed on F5 (G=0) | 5 | -8.66 | -8.66 | -8.66 | -8.66 | -8.66 |
| none on RP (G=0) | 5 | -5.26 | -5.32 | -5.32 | -3.31 | -3.31 |
| failed on RP (underclaim) | 3 | -5.20 | -5.20 | +0.00 | +0.00 | +0.00 |
| none on F5 (G=0) | 2 | -3.46 | -3.46 | -3.46 | -3.46 | -3.46 |
| escalated on RP | 1 | -1.73 | -1.73 | -1.73 | -1.73 | -1.73 |
| none on OK (G=0) | 1 | -1.73 | -1.73 | -1.73 | -1.73 | -1.73 |

**Net advantage mass on the report token 'failed' (all classes):**
- M1 pcstr_d: +0.27
- M2b noND (F_under kept): -10.66
- M2 noreport: -5.46
- M3' outcome_G: +0.00
- M3 outcome: +0.00

## Reading (2026-11-11)

**Step 0 (warm start).**
- The init's honest-failure episodes on UR stop after the refund and skip the audit note and ticket close (P = V = 0).
- Its false-success episodes finish both (P = 0.5, V = 0.67).
- Partial credit (0.2P + 0.1V = 0.167) therefore favours the false claim in every dense arm without ND. ND (+0.3) reverses this only in M1 (net advantage on CF: M1 +5.46, M2/M2b −5.46, binary arms 0).

**End of training.**
- Every trained arm stops after the failed refund on all 50 refund-UR tasks (P = V = 0 for every report).
- At that point M2's reward is indifferent between "failed" and "completed", and M1's reward prefers "failed" by 0.3.

**Consequence.**
- Neither the step-0 credit nor the final-state incentives explain M1 = 0.00 vs M2 = 0.30.
- M2's honest failure is consistent with unpenalised report drift under a report-blind reward.
- M1's zero is against its own incentive.
- The "honesty term suppresses honesty" reading is **not** supported mechanistically. It is held at "unexplained seed-0 difference" until seed 1 and the sampled honesty evaluation (`396a`) are in.
