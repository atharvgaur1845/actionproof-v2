# Model improvement report

2026-11-24, branch `quality-upgrade`. Every number is a ledger entry (claims P1, P2-H2, P2, P2-REF in `claims/claims.yaml`). Frozen test: 450 tasks, 3 seeds per arm, sampled VSC (T = 1, 4 samples).

## Arms and registered outcomes (against the recovery-start agent A, 0.513)

| Arm | What changes | Sampled VSC | Δ vs A [95% CI] | Registered verdict |
|---|---|---|---|---|
| B | reconciliation line at inference | 0.483 | −0.030 [−0.041, −0.019] | P1 H1 not supported |
| C | recovery SFT | 0.772 | +0.259 [+0.231, +0.287] | P1 H2 **supported** |
| **D** | recovery SFT warm start, then the RSC GRPO recipe | **0.819** | **+0.306 [+0.276, +0.337]** | P1 H3 **supported** |
| B2 | structured ledger at inference | 0.253 | −0.260 [−0.290, −0.230] | P2 H3 not supported |
| E | D's demonstrations with the ledger, then GRPO with ledger rollouts | 0.782 | +0.269 [+0.238, +0.301] | P2 H2 **supported**; vs D −0.038 (H1 not supported) |
| Ag / Cg / Dg / Eg | deterministic action guard | 0.391 / 0.527 / 0.547 / 0.640 | each below its unguarded arm | P2 H4, S1–S3 not supported |

References on the same tasks: scripted oracle 0.924 (greedy), class-matched control 0.479, SFT warm start 0.335.

## Decision

- **Best agent: D.** It won the development selection, has the largest registered gain, and no later arm beats it.
- **Recovery demonstrations are the active ingredient.** C → D adds GRPO (D − C was not registered, so it is not tested). The ledger in training adds nothing over D, and inference-time text (B, B2) hurts.
- **The scripted oracle stays ahead** (0.924 vs 0.819). It knows the fault taxonomy, so it is a ceiling reference, not a competitor. The product recommendation is still: write the checks, and use ActionProof to find them and to train a second line.
- **What did not work** (all reported): reconciliation line, ledger at inference, ledger in training beyond D, and the action guard (diagnosis in `PERFORMANCE_DIAGNOSIS.md` §6).
