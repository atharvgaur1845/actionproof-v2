# Research decision: shared-hidden-state theory of group normalisation (Phase II)

Branch `research/latent-state-theory` (local, not pushed). Evidence labels:
- **[math]** proved, with exact-enumeration checks;
- **[sim-reg]** registered held-out analytical simulation (TH1);
- **[LLM-dev]** trained-agent development logs, descriptive;
- **[conf]** confirmatory;
- **[expl]** exploratory.

## 1. What is the strongest proven theorem?
**[math] Theorem 1 (per-state scale invariance).**
- **Statement.** When every rollout of a group shares the hidden environment state, GRPO's expected update
  (ε → 0) is invariant to any per-state positive affine transform of the return, for any policy class and trajectory
  structure.
- **Consequences:**
  - (a) Corollary 1, an exact finite-G stake-free vote for a binary decision with deterministic within-state returns;
  - (b) severity across hidden states provably cannot affect GRPO's update;
  - (c) Corollary 2 (endogenous homogeneity): when reads remove the state, the always-verify corner is governed by
    the vote under **every** exchangeable group law.
- **Companion Proposition 2.** Centred estimators (Dr. GRPO, RLOO) give (G − 1)π(1 − π)ΔQ under every group law.
- **Files:** `docs/theory/PROOF_AND_ASSUMPTIONS.md`, `tests/theory` (44 tests).

## 2. What is genuinely new?
**Within the project:**
- The per-state affine invariance, the large-G 1/σ mixture and the GDPO and AWRB votes were derived in v2
  (`GRPO_FAILURE_DERIVATION.md` Lemma 1, Prop. 1, §3c, §3e).
- The finite-G vote was derived in the lab phase.
- This phase re-proved them independently, delimited them, and added the items below.

**Relative to the literature:**
These are new as far as the searches show (`docs/theory/RELATED_WORK_MATRIX.md`):
- treating the group key in agentic RL as (prompt, hidden environment state), and deriving its exact consequence for
  a costed information action;
- the exact scope of "verify iff p·s > 1/2";
- the π-free Bernstein representation and corner analysis;
- endogenous homogeneity: independent state resampling does not cure GRPO; interior equilibrium, Θ(1/G) regret
  (numerical to G = 64);
- the family-dependent critical correlation (mixture Θ(G^{-1/2}); Dirichlet-multinomial → constant), and the
  insufficiency of pairwise correlation.

## 3. What is already established by prior work?
- 1/σ per-outcome weighting (Bereket and Leskovec 2025).
- Per-prompt monotone transform of success, arcsin√ (Davis and Recht 2025).
- The binary std identity (Bay and Yearick 2026).
- z-score invariance to the shaping scale in all-fail groups (Wang 2026, Dark Room Prop. 1).
- Difficulty bias and std removal (Liu et al. 2025, Dr. GRPO).
- The policy-gradient theorem behind "Dr. GRPO ∝ ΔQ".
- **Verdict on novelty: incremental but defensible.**

## 4. Which assumptions are essential?
- **Shared state within the group:** for the full vote.
- **Aliasing** (A1): the decision does not see z.
- **Deterministic within-state returns:** for the exact vote and the p·s > 1/2 rule. Repairability must be part of
  the state, with no exogenous noise.
- **Reads neutralise the state:** for Cor. 2.
- **c between ε/σ and the stake.**

If repair success is rollout noise (the LLM case), the threshold depends on c (0.654 / 0.667 / 0.706 at s = 0.6).
With enough unrelated within-state noise (q₀ = 0.3), GRPO reaches the Bayes action again.

## 5. Does the theorem predict actual language-agent behaviour?
**Partly, development level.** [LLM-dev] (`results/research/llm_mechanism.json`)
- **Preconditions hold:** 800/800 groups per run share one fault plan, and most clean groups vary only through reads.
- **Per-state invariance holds on real rollouts:** a 20× counterfactual cost change moves GRPO's clean-group update
  by 7%; the centred update moves 13.5×.
- **Direction:** all 7 std-normalised runs move in the direction of the vote estimated from their first 20 steps.
- **Not predicted:** Dr. GRPO's rise in reading is **not** predicted by the warm-start ΔQ (≈ 0); ΔQ became positive
  later in all 10 centred runs (co-adaptation, [expl]).
- **Confirmatory:** pending AD-C1.

## 6. Does AD-C1 support the registered hypothesis?
**Pending.** The theory's predictions T1–T8 were frozen before any cw_conf output
(`docs/research/AD_C1_THEORY_PREDICTION.md`, commit 2354929). The registered statistic is the lead engineer's.

## 7. Does the result generalise beyond the existing task distribution?
- **The mechanism, on synthetic problems: yes [sim-reg].** TH1 held-out, 731 cells:
  - H1–H5 supported: decision prediction 721/721, vote 152/152 including 77/77 where vote and ΔQ disagree,
    median π_T error 0.0003;
  - H6 not supported: GRPO slow under large stakes;
  - H7 untestable.
- **Other agent systems: not established.** It is argued from their group construction only.
- **The learned policies:** the S4 shifts concern the recovery-start agent, not the estimator.

## 8. Is there any new algorithm worth pursuing?
**No.** Four candidates were checked (`docs/research/EXPERIMENTAL_AUDIT.md` §5):
- state decorrelation (provably partial);
- external scales (≈ Dr. GRPO; EMA arms collapsed);
- observation-measurable returns (AWRB: breaks under state-dependent costs or stochastic repair);
- stake weighting (needs hidden-state labels).

The recommendation is the existing fix: drop the group std, or use a scale independent of the group's hidden
states.

## 9. Claims that can safely be made
1. Theorem 1, Cor. 1, Cor. 2 and Prop. 2, with the stated assumptions.
2. The exact threshold table, and the scope of p·s > 1/2.
3. TH1 H1–H5 as a registered held-out analytical simulation. H6 is reported as not supported, with the post-hoc
   reason labelled.
4. In ActionProof's training, groups share the hidden state; the std-normalised update is cost-invariant on real
   rollouts; std-normalised runs follow the early vote (development, 2 seeds per arm plus 2 in-progress C1 seeds).
5. With the database snapshot, LLM judges agree with the verifier, and without it they fail. The policy-family judge
   accepts 70% of failed executions. This is model-vs-verifier evidence only.

## 10. Claims to remove or weaken
1. "GRPO learns p·s > 1/2 irrespective of read costs": restrict to state-level repairability and deterministic
   returns.
2. "Dr. GRPO's update is exactly proportional to the value of information": replace with the on-policy action-value
   gap ΔQ. It equals net VOI only under Bayes-optimal continuation.
3. "494 of 495 cells": these are runs, predicted by the exact enumerated sign, on the development grid.
4. "Condition B weakens the bias": now "reduces it to Θ(1/G) regret for non-rare faults; does not remove it".
5. The novelty framing of "a new mechanism": it is a new consequence of a known one.
6. J1's "Gemma licence" for gemma-4-31B-it: the card says Apache 2.0.
7. main_v2 says "494 of 495 cells" and carries author identity: superseded by main_v3.

## 11. Remaining limitations
- One model family and one simulator.
- The confirmatory LLM test (AD-C1) is pending.
- The exact theory covers one binary decision; multi-decision is covered by Thm 1 (qualitative) and experiments.
- The read proxy counts uninformative reads.
- No human-labelled judge data.
- No LLM run with plain GRPO at another cost, and none with resampled states (V2-E1 never ran).
- H7 was untestable.

## 12. Is the paper ready for submission?
**Not yet.**
- main_v3 builds cleanly (7 pages, main text ends on p. 5) and is anonymised.
- The AD-C1 `\pend{}` must be filled or removed.
- The lead engineer must add the v3 numbers to `claims/claims.yaml` and the canonical pipeline.
- The author must approve the title and thesis.

## Decision: **Interesting result requiring further validation**, leaning to "defensible but incremental"
The mathematics is correct and now delimited, and the registered held-out testbed confirms its quantitative
predictions. But the language-agent link is development-level, one of its predictions (Dr. GRPO from the warm-start
ΔQ) failed, and the single confirmatory test is pending. If AD-C1 meets T1–T3, the paper becomes "defensible but
incremental". If T1 or T2 fails, the paper should be framed as an analytical result with a negative language-agent
test.

## Decisions for the author
1. Adopt main_v3's thesis and title, or an alternative:
   - "Stake-Blind by Construction: GRPO and Costly Verification under Shared Hidden States";
   - "Group Normalisation Cannot See the Stakes of Verification".
   The candidate title "When Group Rollouts Share Hidden States: GRPO's Stakes-Blind Learning of Verification" is
   accurate if "learning of verification" is kept to the binary decision.
2. Accept the narrowed claims in §10.
3. Approve adding the v3 numbers to the claims ledger (lead engineer).
4. Label the 76 human calibration cases, or keep the judge section explicitly model-vs-verifier.
5. Whether to cite CTA's collapse as consistent with the theorem without verifying its group construction (we do not
   claim it).
6. Push `research/latent-state-theory` (requires your approval).

## N. Number → file map (main_v3 and this report)

| Number(s) | File / key |
|---|---|
| thresholds 0.4948 / 0.499 / 0.4998; 0.8246 / 0.8316 / 0.833; 0.654 / 0.667 / 0.706; 0.452, 0.412 | `results/theory/scan.json` → `threshold_exactness`, `cost_blindness` |
| π* 0.597, regret 0.101; G(1−π*) ≈ 2.4 | `scan.json` → `iid_interior` |
| ρ* 0.211 / 0.149 (mix), 0.152 → 0.305 (dirmult) | `scan.json` → `critical_rho` |
| −2.961, −0.0147 → +57.26 | `scan.json` → `severity.rescale_witness` |
| TH1: 721/721, 152/152, 77/77, 67/69, 661/661, 53/75, 3 pairs, 0.0003, agreement/regret table | `results/theory/th1_heldout_summary.json` |
| H6 post hoc: 22 cells, gap 0.062 vs 0.010, regret 0.207 vs 0.011, max error 0.045 | `results/theory/th1_heldout_posthoc_h6.json` |
| 800/800, 111/146 …, −898 → −962, −86 → −1163, votes, read shares, ΔQ windows | `results/research/llm_mechanism.json` |
| P1 +0.259 / +0.306 / −0.030 | `results/p1/p1_summary.json` |
| judge F1 0.990, false acceptance 0.700, Aya fooled 0.875, cost $2.816 | `results/j2/j2_summary.json` |
| importance ESS 1.01 / 322 | `results/quality/g_mismatch_s0_ck100.json` |
| TB1 149/159, 319/495, 28/495 | `results/ad2/tabular/TB1.jsonl.gz` (recount), `results/lab/exact_analysis.json` |
