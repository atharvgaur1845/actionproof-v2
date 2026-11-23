# Generalisation and robustness results

All results are development evidence. Testbed numbers come from `results/ad2/tabular/TB3_summary.json` and
`TB4_summary.json` (5 seeds each); LLM numbers come from the first-stage result files named.

## 1. Testbed TB3: does the learned rule transfer? (feature policy; 5 seeds)

**Setup.** The policy sees only risk descriptors (logit p, log s, log c, …). It was trained on 24 tasks over the
grids p {0.05, 0.2, 0.6}, s {0.5, 1}, c {0.01, 0.05, 0.2}. Columns:
- **agree:** the share of operations whose learned decision has the sign of the exact VOI (|VOI| > 0.02);
- **regret:** J* − J, exact.

| Estimator / budget | Episodes | test_iid regret / agree | test_shift (unseen p, s, c) regret / agree | test_long (unseen length) regret / agree |
|---|---|---|---|---|
| critic_q ×4 | 51,200 | 0.0039 / 0.97 | **0.0038 / 0.98** | 0.0291 / 0.88 |
| igc ×4 | 51,200 | 0.0047 / 0.97 | 0.0041 / 0.97 | 0.0335 / 0.90 |
| nostd ×4 (= rloo ×4) | 51,200 | 0.0058 / 0.97 | 0.0045 / 0.98 | 0.0295 / 0.89 |
| ctxema ×4 | 51,200 | 0.0056 / 0.97 | 0.0048 / 0.97 | 0.0359 / 0.88 |
| cvc | 55,467 | 0.0120 / 0.97 | 0.0100 / 0.97 | 0.0623 / 0.85 |
| fork_bel | 64,000 | 0.0130 / 0.97 | 0.0103 / 0.97 | 0.0633 / 0.85 |
| fork_act | 64,000 | 0.0153 / 0.97 | 0.0118 / 0.97 | 0.0699 / 0.82 |
| igc ×1 | 12,800 | 0.0160 / 0.97 | 0.0114 / 0.97 | 0.0681 / 0.92 |
| critic_q ×1 | 12,800 | 0.0154 / 0.97 | 0.0120 / 0.97 | 0.0711 / 0.81 |
| awrbn | 12,800 | 0.0185 / 0.97 | 0.0136 / 0.97 | 0.0766 / 0.77 |
| nostd ×1 | 12,800 | 0.0201 / 0.97 | 0.0151 / 0.97 | 0.0828 / 0.70 |
| ctxema ×1 | 12,800 | 0.0464 / 0.90 | 0.0267 / 0.88 | 0.1259 / 0.62 |
| **grpo** | 12,800 | 0.3133 / 0.56 | 0.1830 / 0.54 | 0.2079 / 0.74 |

**Readings.**
1. **A transferable decision rule is learnable by every correct-sign estimator.** Agreement with the VOI sign on
   unseen fault rates, repair rates and costs is 0.97–0.98. GRPO's 0.54 is near chance. The rule is learnable here
   because the policy class can represent the VOI boundary (it is linear in log descriptors). That is a property
   of the representation, not of any estimator.
2. **Unseen workflow length is the hard case for everyone** (agreement 0.70–0.92). The VOI of a read depends on
   whether the other writes succeed (multiplicative success), which the descriptor features do not encode.
3. **At matched budget, forks lose.** Every fork estimator (cvc, fork_bel, fork_act; 55–64k episodes) has 2–3×
   the regret of cheap estimators trained 4× longer (51k episodes). At equal *update* counts the forks help:
   cvc 0.0100 vs nostd ×1 0.0151 on test_shift.
4. **nostd and rloo are identical.** Under Adam, RLOO's G/(G − 1) rescaling of the nostd advantage has no effect.

## 2. Testbed TB4: misspecified hidden-fault priors (rich, multi-fault, c = 0.05, L_f = 1; 5 seeds)

Mean regret over the three workflows (sd over seeds in brackets). The safe-op verify rate is shown where it
matters.

| Estimator (information) | true | est (empirical Bayes) | ×4 rates | ×0.25 rates | flat 0.3 (safe op wrongly faultable) |
|---|---|---|---|---|---|
| nostd (none) | 0.0347 (0.0103) | — | — | — | — |
| fork_act (snapshot only) | 0.0156 (0.0021) | — | — | — | — |
| igc (prior support) | **0.0121** | 0.0328 | 0.0121 | 0.0121 | 0.0328 (gating lost; safe-op verify 0.37) |
| cvc (snapshot + prior) | **0.0073** | 0.0096 | 0.0062 | **0.2536** (stops verifying) | 0.0545 (safe-op verify 0.995) |
| fork_bel (snapshot + prior) | 0.0145 | 0.0168 | 0.0112 | **0.2558** | 0.0609 |
| awrbn (simulator + prior) | 0.0295 | 0.0370 | 0.0496 | **0.2604** | 0.0670 |
| awrb (simulator + prior; group std) | 0.0089 | 0.0101 | 0.0101 | 0.0563 | 0.0563 |

**Readings.**
1. **Under-estimated fault rates are catastrophic for every belief-weighted estimator** (cvc, fork_bel, awrbn).
   With p̃ = p/4, the belief makes reading look worthless, the policy stops reading, and no read ever reveals the
   error. The failure is self-confirming. This is the "exact hidden-fault distribution" deployment limitation the
   assignment asks about, and it is severe.
2. **The empirical-Bayes prior removes it** (cvc/est 0.0096). The prior is estimated only from what reads
   revealed, so it is self-correcting as long as some reads happen.
3. **igc needs only the support.** It is unaffected by wrong *weights* (×4, ×0.25), but loses its advantage when
   the support is wrong (flat 0.3 makes the never-faulted op look faultable). Its `est` variant equals `flat`
   because the Beta(1, 1) start puts mass on every op type, including the safe one: an implementation property
   of `assumed_cfg("est")`.
4. **Over-estimated fault rates are mildly helpful here** (cvc ×4: 0.0062), because verification is mostly
   worthwhile in this configuration. This is not a general result.
5. **awrb (group std) is robust to ×0.25 where awrbn is not** (0.056 vs 0.260). The mirror bias pushes it to
   verify regardless. That is the same bias that makes it over-verify when reading is worthless (TB1).

## 3. LLM development evidence (first stage, cw_dev / proc_dev, greedy; context only)

| Shift | Evidence | Status |
|---|---|---|
| Different task split (proc_dev: free-read standard split, a different objective) | VSC: grpo 0.487/0.487, nostd 0.667/0.653, awrbn 0.600/0.600, awrb 0.793/0.793 (seeds 0/1, `results/ad/dev/*__proc_dev.summary.json`) | descriptive |
| Different verification cost (c = 0.30, seed 0) | awrb and awrbn both stop reading entirely (VSC 0.707). Consistent with VOI < 0 at most sites at c = 0.30 | descriptive; cannot separate rules (first-stage F6) |
| Renamed APIs (s4_api), Hindi (s4_lang_hi), unseen 4-step workflow (s4_wf4), capability retention (R1) | Registered in AD-C1 §4, run by the C1 evaluation tasks after C1 training | **pending** (FINAL_ALGORITHM_REPORT §5) |
| Hinglish, cross-domain (OpsSim), natural-language reformulations | not run for the v2 programme (GPU reprioritisation, DV-2) | **not tested** |

No v2 method was trained on the LLM. LLM generalisation of any v2 candidate is therefore **untested**.
