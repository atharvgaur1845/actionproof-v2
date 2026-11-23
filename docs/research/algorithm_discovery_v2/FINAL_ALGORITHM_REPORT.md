# Final report: second-stage algorithm discovery (v2)

Branch `research/algorithm-discovery-v2`. All numbers are taken from the files named. Development and confirmatory
evidence are kept separate.

## 1. Bottom line

1. **The GRPO failure mechanism is established mathematically and reproduced:** exactly in a controlled testbed, and
   descriptively in the LLM training logs.
   - Under group-relative std normalisation, a verification cost cancels inside every group where verification
     changes only cost. The policy then verifies iff P(verification changes the outcome) ≳ 1/2. Cost, task value
     and failure penalty play no role (the "vote rule").
   - The precise condition is broader than "shared hidden world": **any advantage scale estimated from samples
     whose hidden-state composition varies** distorts the decision. That covers group std (shared or resampled
     worlds at finite G), batch std over few groups, per-node or fork-group std, and GDPO's per-component std.
   - A scale independent of the current samples' hidden states does not distort it: none, RLOO, or EMA of past
     scales.
2. **AWRB, the first-stage candidate, is not a better decision-making estimator.**
   - Its reported VSC gain over no-std GRPO (0.867 vs 0.833 on cw_dev) comes from the group-std variant's mirror
     bias, i.e. over-verification: J_c is lower in both seeds, and it verifies the never-faulted journal 91% of
     the time.
   - The unbiased variant (awrbn) is below no-std GRPO in the LLM and about equal in the testbed.
   - It needs privileged information: the simulator and the prior.
   - It fails catastrophically when fault rates are under-estimated.
3. **No new algorithm with confirmatory evidence was discovered.** Of six candidate families, one survives development
   gates in the testbed:
   - **information-gated exact local credit (igc).** An uninformative verification decision gets its exact value −c
     instead of the sequence-level advantage.
   - It passes the registered gates at 1× and 4× budget against the strongest budget-matched baseline. In TB2 it
     cuts never-needed verification about 10× at 1× budget (0.197 → 0.020).
   - It does not pass at 8× budget: its benefit is sample efficiency, not a different end point.
   - Its LLM version (first-stage D3) had not run at the time of writing.
   - It is per-decision credit with an exact local value. It is not novel as a mechanism.
4. **Forked counterfactual credit** (actual-world or belief-resampled forks) is unbiased and best per update, but it
   is **not cost-effective** at matched episode budget (registered gate failed in 5/5 seeds).

## 2. Answers to the ten research questions

**1. Is the GRPO failure mechanism mathematically established and empirically reproduced?** Yes, with a corrected
condition.

| Evidence | Result |
|---|---|
| Derivation | GRPO_FAILURE_DERIVATION §2–5 |
| Numerical demonstration | grpo threshold p = 0.50–0.52 for every c ∈ {0.01, 0.05, 0.2} at s = 1; VOI threshold c/s |
| TB1 | grpo c-insensitive in 97% of cells, wrong decision in 64%, vs nostd 5.7% |
| TB6 | insensitive to G, warm start and optimiser |
| LLM decomposition | reproduced exactly: −143.6 / −0.36, or −134.3 with TRL's own denominator. Faulted groups split +49.6 (outcome) / −49.6 (cost-only), the vote rule observed directly |

Correction to the first stage: resampling the hidden world per rollout does **not** remove the distortion at finite
G. Registered H1c failed: 0.315 wrong. Agreement rises 0.53 → 0.74 → 0.98 for G = 4 → 8 → 16.

**2. Is it distinguishable from existing GRPO-normalisation literature?** Partly.
- **Not ours:** the 1/σ weighting (Bereket and Leskovec 2025) and the no-std fix (Dr. GRPO).
- **Ours, after the searches in LITERATURE_AND_NOVELTY_AUDIT:**
  - the consequence for information-gathering actions (the vote rule, cost-blind thresholds);
  - the normaliser-independence condition and its predictions for batch std, fork/tree-node std, GDPO and
    constrained GRPO (all confirmed in TB1/TB2);
  - the counterexample: other within-group variance restores cost sensitivity.

  We found no prior paper with these, but absence of evidence from four searches is weak.
- **Consistent outside report:** Calibrate-Then-Act (arXiv 2602.16699) observed GRPO-trained agents collapsing to a
  static exploration policy. That is not verified to be this mechanism.

**3. Does AWRB beat no-std GRPO across independent seeds?**

| Variant | Development result (cw_dev, 2 seeds) | Testbed |
|---|---|---|
| awrb (group std) | VSC +0.034 in both seeds; J_c lower in both | — |
| awrbn | VSC −0.060 and J_c −0.025 in both seeds | ≈ nostd (TB1 0.0123 vs 0.0132; TB2 gate failed at every budget) |

The confirmatory test is H_AWRB = awrbn > nostd in C1; awrb vs nostd is not a registered confirmatory hypothesis.
See §5.

**4. Does AWRB require privileged simulator information?** Yes: an exact deterministic simulator, the support and
weights of the hidden-fault prior, and counterfactual final states. It is valid as a training-time privileged
estimator; deployment needs nothing.

**5. Does it remain effective under prior misspecification?** No. With fault rates under-estimated ×0.25, awrbn
regret is 0.260 vs 0.030 with the true prior (TB4): it stops verifying, so no evidence corrects the belief. The same
failure hits every belief-weighted method (cvc 0.254, fork_bel 0.256). An empirical-Bayes prior estimated from
revealed reads removes it (cvc 0.0096). igc needs only the support, so wrong weights do not affect it (0.0121 under
true, ×0.25 and ×4); it degrades when the support is wrong.

**6. Has any genuinely new algorithm been discovered?** No genuinely new mechanism. The surviving candidate (igc) is a
specific, exact, per-decision credit rule built from known parts: per-step credit, all-action gradients, and
replay-based uninformativeness. It is supported by development evidence in a tabular testbed only.

**7. Does it outperform all relevant baselines?** In the testbed, at 1× and 4× budget, igc beats every non-privileged
baseline:
- nostd, RLOO, learned V and Q critics, EMA scales and Lagrangian;
- paired regret difference vs nostd at 1×: +0.0141, CI [+0.0117, +0.0167], d_z 1.45, Holm p < 1e-4.

At 8× it is within 0.001 of nostd and RLOO (Holm p 0.18–0.29). It is untested in the LLM.

**8. Does it learn selective verification rather than unconditional reading?** In the testbed, yes:
- never-faulted-op verify 0.020 vs nostd's 0.197 at 1× budget;
- uninformative reads 5.2% vs 16.3%;
- risky-op verification and recovery are not reduced.

In the LLM no arm has yet been shown selective (first-stage F3).

**9. Does the benefit generalise?**
- **Testbed (TB3):** every correct-sign estimator, igc included, learns a decision rule that transfers to unseen
  fault rates, repair rates and costs (agreement with the VOI sign 0.97–0.98). igc ×4 has test_shift regret 0.0041,
  vs nostd ×4's 0.0045. Unseen workflow length is hard for every method (agreement 0.70–0.92).
- **LLM:** not tested for any v2 method. The first-stage shift evaluations (renamed APIs, Hindi, unseen workflow)
  and capability retention run in the C1 evaluation tasks (§5).

**10. Strongest defensible contribution to the submission?**
- **(a)** The diagnosis: the vote rule, the normaliser-independence condition, and their exact testbed confirmation
  across 18 estimators, including the predictions that batch std, GDPO, constrained GRPO and per-node std fail.
  Reproduced in the LLM logs.
- **(b)** The AWRB audit: an honest negative. Its apparent gain is over-verification, it needs privileged
  information, and it is fragile to prior misspecification.
- **(c)** A development-stage result that exact local credit for provably uninformative verification decisions makes
  selectivity about 4× more sample-efficient in the testbed. Its LLM evidence is pending.

**Do not claim a new algorithm.**

## 3. Specification of the surviving candidate (igc), for completeness

**Information needed at training time.** A simulator that can replay recorded calls in any world of the hidden-fault
prior's **support**. Weights are not needed. Nothing is needed at deployment.

**Uninformativeness test.** For a verification decision at history h, with Z(h) the set of support worlds consistent
with h, the decision is uninformative iff the verify observation o_v(h, z) is the same for every z ∈ Z(h). Then the
read cannot change any later decision, and its exact action-value difference is

  Q(h, verify) − Q(h, skip) = −c,

for every continuation policy. The LLM implementation of this test is `actionproof/ad/readinfo.py`.

**Update.** For such decisions, use the all-action term

  g_t = Σ_a ∇π(a | h_t)·Q(h_t, a) = π(1 − π)·(−c)·∇logit(verify | h_t),

independent of the sampled action and of the episode outcome.

For all other decisions, use the sequence-level advantage *without* group std (R_i − mean_g R). Any scale must be
independent of the current samples' hidden states.

**Unbiasedness.** The exact value replaces a Monte Carlo estimate of the same quantity at those decisions; the
remaining decisions use the unbiased no-std estimator. The combination is an unbiased policy gradient, because the
per-decision terms are exact.

**Why selectivity follows.** Under sequence-level credit, a worthless read and a valuable read in the same
trajectory share one advantage. The only signal separating them is c per read, between trajectories, buried in
outcome noise. igc delivers that signal exactly, at every uninformative decision, without noise.

**Limitations.**
- The benefit is bounded by the share of uninformative decisions. In LLM cw_dev, 42–49% of all reads were
  uninformative (`results/ad/read_waste_cw_dev.json`).
- It needs the support to be right (TB4 flat:0.3).
- It does nothing for informative-but-unprofitable reads (p·q·L < c). Those still rely on the noisy sequence signal.
- In the LLM, D3 combines it with the EMA scale, which collapsed in 2 of 4 D2 runs (V-F8).

## 4. Required outputs: where everything is

| Output | File |
|---|---|
| Derivation | GRPO_FAILURE_DERIVATION.md |
| AWRB audit | AWRB_IMPLEMENTATION_AUDIT.md |
| Literature and novelty | LITERATURE_AND_NOVELTY_AUDIT.md |
| Controlled testbed | CONTROLLED_TESTBED.md |
| Candidates | ALGORITHM_CANDIDATES.md |
| Registry | EXPERIMENT_REGISTRY.csv |
| Smoke and experiment log | SMOKE_EXPERIMENT_LOG.md |
| Failures | FAILURE_ANALYSIS.md |
| Deviations | DEVIATIONS.md |
| Selective verification | SELECTIVE_VERIFICATION_RESULTS.md |
| Generalisation | GENERALISATION_RESULTS.md |
| Ablations | ABLATION_RESULTS.md |
| Statistics | STATISTICAL_ANALYSIS.md |
| Figures | `figures/` |
| Code | `actionproof/ad2/` |
| Scripts | `scripts/ad2/` |
| Tests | `tests/ad2/` |
| Raw testbed results | `results/ad2/tabular/*.jsonl.gz` |
| Testbed summaries | `results/ad2/tabular/*_summary.json` |

**Reproduction:** CONTROLLED_TESTBED.md §3.

**Checkpoints:** none from v2. The tabular policies are small and written in each run's JSON line (`theta`, for
non-core runs).

## 5. AD-C1 confirmatory verdict (first stage)

PENDING. This section is filled from `results/ad/c1_summary.json` only, after tasks 1220–1231 (training) and
1240–1241 (evaluation) finish and `scripts/ad/c1_stats.py` runs once.

**Arms.** grpo, awrb, nostd and awrbn. D2 arms failed their gates. D3 seeds 2–4 cannot be trained under the GPU
reprioritisation (PREREGISTRATION_AD_C1_NOTE3).

**Prediction recorded before any cw_conf output.** H_AWRB (awrbn > nostd) is not supported (AWRB audit §4).
