# Theorem audit: is the shared-hidden-state result correct, and what exactly does it say?

Auditor: research agent on `research/latent-state-theory`, 2026-11-24. The theorem was reconstructed from the code and
re-derived independently (`docs/theory/PROOF_AND_ASSUMPTIONS.md`). The previous derivation
(`docs/research_lab/GRPO_MECHANISM_FORMAL.md`) was used only for comparison.

**Verdict.** The previous Theorem 1 is **correct under its stated assumptions.** The mission statement's summary
of it overreaches in three places, listed in §1. No error was found that invalidates a published number. Two numbers
are mislabelled (§2).

## 1. Statement-level findings

| # | Claim as circulated | Finding | Severity |
|---|---|---|---|
| F1 | "GRPO learns the threshold p·s > 1/2" | **Exact only when repairability is part of the shared hidden state** (states repairable / unrepairable), and within-state returns are deterministic. In TB1 and in the LLM, repair success is rollout randomness. The exact threshold is then different and cost-dependent: 0.654 / 0.667 / 0.706 at s = 0.6, c = 0.01 / 0.05 / 0.2, against 1/(2s) = 0.833 (`results/theory/scan.json` `threshold_exactness`) | must narrow |
| F2 | "irrespective of read costs" | True iff each state's cost is positive and below that state's stake, c ≫ ε/σ̂ (c ≳ 0.005 at ε = 1e-4), and returns are deterministic within a state. False when c exceeds a stake (that state's vote flips), at c ≲ 10⁻³ (p* = 0.452 at c = 0.001), and with any within-state noise. Exogenous failure q₀ = 0.3 makes GRPO choose the Bayes action again (scan `counterexamples/exogenous_noise`) | must narrow |
| F3 | "Dr. GRPO's expected update is exactly proportional to the value of information" | It is exactly (G − 1)π(1 − π)·ΔQ, where ΔQ = Q(verify) − Q(skip) **under the policy's own downstream behaviour**, for every group law (Prop. 2.2). ΔQ equals decision-theoretic net VOI only if the post-read behaviour is Bayes-optimal. During LLM training it is not: ΔQ̂ was ≈ 0 at the warm start and became positive only after step 20 (`results/research/llm_mechanism.json`). Proportionality also needs one binary decision; with several decisions the general statement is (G − 1)∇J | must narrow ("on-policy action-value gap", not "VOI") |
| F4 | "the vote" | Correct as stated (finite G, exact, ε → 0). It is a corollary of a more general and simpler fact: under a shared state, GRPO's expected update is invariant to any per-state positive affine rescaling of the return, for any policy and trajectory structure (Theorem 3.1) | strengthen and simplify |
| F5 | "494 of 495 cells" | These are **runs** (165 cells × 3 seeds), predicted by the exact enumerated sign at π₀ = 0.3 with s as rollout noise, not by p·s > 1/2. Cell level: the plain p·s > 1/2 rule agrees with GRPO's TB1 decision in 149/159 cells (0.937). The cell filter (c ≥ 0.01, \|VOI\| > 0.02) on the closed form gives 159 cells; the record-level VOI filter gives 165 (rounding at \|VOI\| = 0.02) | relabel |
| F6 | Prediction is "independent validation" | TB1 is the grid that motivated the theorem, so it is development evidence. The independent held-out test is TH1 (`PREREGISTRATION_TH1.md`) | reclassify |

## 2. Code-to-math audit

| Item | Code | Matches the math? |
|---|---|---|
| Group construction | TRL `num_generations=8`, one prompt per task, one fault plan per task (`actionproof/ad/pools.py`) | yes; 800/800 groups per logged run share one plan (`llm_mechanism.json`) |
| Std | `nanstd`, Bessel correction, + 1e-4 | yes (exact engine uses the same) |
| Loss | TRL default `loss_type="dapo"`, `num_iterations=1` | so "nostd" = Dr. GRPO's advantage with DAPO token aggregation; ratio = 1, clipping inactive |
| Environment determinism | ids only (hashed in `CWEnv`) | yes; within-group variation is policy-only |
| Reward | R_V − 0.05·n_verify, n_verify counts all read-tool calls | reads after the journal entry are cost-only; they enter the vote as cost-only states |
| `exact_vote.py` | brute-force multinomial | agrees with the new Bernstein engine to 1e-9 on 48 parameter/estimator/law combinations (`tests/theory`) |
| TB1 recount | `scripts/lab/exact_analysis.py` | reproduced: GRPO 319/495 runs wrong, Dr. GRPO 28/495 |

## 3. What survived the falsification attempts

- Every identity in PROOF_AND_ASSUMPTIONS was checked by exact enumeration. The tests assert both the identity and
  its failure where the theory says it should fail: affine invariance fails under iid; pairwise correlation does not
  determine the update for G ≥ 3.
- The independent sampled-group trainer reproduces the mean-field predictions. Dev split: median |sim − pred| ≤ 0.0003
  (`results/theory/th1_dev_summary.json`). Held-out: `results/theory/th1_heldout_summary.json`
  (TESTBED_RESULTS.md).
- Real rollouts: re-weighting fixed GRPO training rollouts at a counterfactual cost from 0.05 to 1.0 changes the
  GRPO clean-group update by 7% (−898 → −962) while the centred update grows 13.5× (−86 → −1163). This is the
  predicted per-state invariance on real data.

## 4. What did not survive, or is not supported

- The claim that Dr. GRPO's LLM behaviour is "predicted by VOI": the warm-start ΔQ̂ did not predict the increase in
  reading (s0: −0.082 [−0.163, −0.014]; s1: +0.023 [−0.030, +0.069]). It is consistent with a co-adapting ΔQ only
  (exploratory).
- Resampling states independently does **not** cure GRPO (Cor. 3.3). Earlier documents described condition B as
  "weaker but present"; it is now explained. Near always-verify the read-neutralised returns make groups
  homogeneous, so the vote returns under every group law. At G = 8, p = 0.3, c = 0.05, GRPO stalls at π* = 0.597.
- AWRB's sign guarantee fails with state-dependent read costs (scan `state_dependent_cost_b`: regret 0.017) and with
  stochastic repair (over-verification). The v2 18/18 over-verifying mirror cells have s < 1 by definition (§5).

## 5. Open items

1. **Resolved.** The v2 "mirror cells" are defined by VOI = p·s(1 + L) − c < 0 < p(1 + L) − c
   (`docs/research/algorithm_discovery_v2/ABLATION_RESULTS.md` L45). That implies s < 1, so Prop. 4.2's explanation
   applies by construction. Prop. 4.2's clean-group vote ("read iff p(1 + L) > c, independent of repair") was
   **already derived** in v2 (`GRPO_FAILURE_DERIVATION` §3c, cited in `AWRB_IMPLEMENTATION_AUDIT.md` L147). It is
   prior internal work, not a new result of this phase.
2. The large-G statement (Prop. 3.6) is proved by the law of large numbers plus Slutsky. It is not checked numerically
   for G > 64.
