# Ablation results

These are development results from the controlled testbed. Agreement = the fraction of cells whose learned verify
decision has the sign of the exact VOI (|VOI| > 0.02, c ≥ 0.01). The analysis is `scripts/ad2/tab_analyze.py`.

## 1. Mechanism factors (TB6: core workflow, s = 1, p {0.2, 0.4, 0.6} × c {0.01, 0.05, 0.2}, 3 seeds)

Mean agreement, marginalised over the other factors (`results/ad2/tabular/TB6_summary.json`):

| Factor | grpo, A (shared world) | grpo, B (resampled) | nostd, A | nostd, B |
|---|---|---|---|---|
| warm start ρ0 = 0.05 / 0.3 / 0.6 | 0.27 / 0.38 / 0.38 | 0.74 / 0.75 / 0.76 | 0.94 / 1.00 / 1.00 | 0.95 / 1.00 / 1.00 |
| group size G = 4 / 8 / 16 | 0.31 / 0.35 / 0.35 | **0.53 / 0.74 / 0.98** | 0.97 / 0.98 / 0.99 | 0.97 / 0.99 / 1.00 |
| optimiser Adam / SGD | 0.34 / 0.34 | 0.76 / 0.74 | 1.00 / 0.96 | 1.00 / 0.97 |

**Readings.**
- **Condition A.** GRPO's distortion is insensitive to group size, warm start and optimiser: the vote rule.
- **Condition B.** It shrinks exactly as the normaliser-independence condition predicts. The share of
  cost-only (all-clean) groups falls as (1 − p)^G.
- **Warm start.** It matters only at ρ0 = 0.05 (slower movement, about √(ρ(1 − ρ))). It does not change the sign.
- **Optimiser.** It matters only marginally. Adam's normalisation does not undo a world-dependent per-group scale.

## 2. Advantage normaliser (TB1: 216 cells, 3 seeds)

| Estimator | What differs from nostd | Wrong-decision rate | Mean regret (c ≥ 0.01) |
|---|---|---|---|
| nostd (Dr. GRPO) | — | 0.057 | 0.0132 |
| rloo | leave-one-out baseline | 0.057 | 0.0132 (identical under Adam) |
| ema (global EMA scale; first-stage D2) | world-independent scalar | 0.048 | 0.0110 |
| ctxema (per-context EMA scale) | per-task world-independent scale | 0.071 | 0.0127 |
| batchstd (4 groups) | batch-level std | 0.366 | 0.0480 |
| grpo_B | group std, resampled worlds | 0.315 | 0.0440 |
| grpo | group std | 0.644 | 0.1388 |
| lagr_grpo | group std + Lagrangian cost | 0.628 | 0.1235 |
| gdpo | per-component group std | 0.747 | 0.1880 |

**Readings.**
- Every normaliser computed from samples whose hidden-state composition varies is biased: grpo, grpo_B, batchstd,
  gdpo, lagr_grpo.
- Every normaliser independent of the current samples is not: none, rloo, ema, ctxema.
- The Lagrangian multiplier does not rescue group-std GRPO (0.628 vs 0.644), as derived.

## 3. Return estimator (TB1)

| Estimator | Wrong rate | Regret | Mirror cells, verify fraction (n = 18; VOI < 0 but p(1 + L_f) > c) |
|---|---|---|---|
| awrb (RB return + group std) | 0.091 | 0.0192 | **1.00** |
| awrbn (RB return, no std) | 0.042 | 0.0123 | 0.00 |
| nostd (raw return) | 0.057 | 0.0132 | 0.00 |

The mirror bias of group-std AWRB is confirmed: it verifies in every cell where reading is worthless but
p(1 + L_f) > c. Rao-Blackwellising the return (awrbn vs nostd) lowers regret slightly in the tabular testbed
(0.0123 vs 0.0132). In the LLM it did not help (AWRB audit §0).

## 4. Decision-level credit (TB1; not budget-matched; budget matching is TB2's)

| Estimator | Extra information | Wrong rate | Regret | Safe-op verify (mean over all c) |
|---|---|---|---|---|
| cvc | snapshot + prior | **0.012** | **0.0043** | 0.161 |
| igc | prior support | 0.034 | 0.0060 | 0.154 |
| fork_bel | snapshot + prior | 0.032 | 0.0060 | 0.161 |
| fork_act | snapshot | 0.051 | 0.0072 | 0.158 |
| critic_q | none | 0.053 | 0.0124 | 0.293 |
| critic_v | none | 0.059 | 0.0163 | 0.317 |
| nostd | none | 0.057 | 0.0132 | 0.299 |
| fork_bel_std | snapshot + prior, fork-group std | — | see TB2 | — |

The safe-op rate averages over c = 0 cells, where verifying the safe op is free, so values far from 0 are
expected.
