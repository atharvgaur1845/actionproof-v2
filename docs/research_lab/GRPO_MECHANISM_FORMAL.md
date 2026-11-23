# The cost-blind group vote: a formal reconstruction

Status: Theorem 1 and its corollaries are proved below. They are checked exactly by enumeration
(`actionproof/lab/exact_vote.py`, `tests/lab/test_exact_vote.py`, 21 tests). Numbers come from
`results/lab/exact_analysis.json` (`python scripts/lab/exact_analysis.py`).

Relation to prior work: the 1/σ weighting is Bereket and Leskovec (arXiv 2508.11800); dropping the std is Dr. GRPO
(arXiv 2503.20783). For binary rewards, Bay and Yearick (arXiv 2607.00152, Theorem 1) show that GRPO's scalar update
equals the group's reward std σ = √(k(G−k))/G, and Dr. GRPO's equals σ²·Δs. Theorem 1 below is their identity
carried over to any two-valued return (any stakes) and mixed over hidden states. The part we add is what the mixture
does to a decision whose payoff depends on a hidden state: the result is a stake-free vote across states.

## 1. Setting

- **Task and latent fault.** A task x has a hidden execution state z ~ π₀(·|x): which writes failed silently, were
  applied partially, were duplicated or timed out. The policy never observes z. A write's "ok" acknowledgement is the
  same in every z.
- **Verification action.** After an acknowledgement, the agent chooses r ∈ {0 continue, 1 verify}. It verifies with
  probability ρ = π_θ(verify | h), with logit θ. The score is ∂ log π(r)/∂θ = r − ρ.
- **Cost and payoffs.**
  - Verification costs c > 0.
  - A detected fault is repaired with probability s (the recovery probability).
  - Completion is worth V. Failure costs L_f.
  - Exogenous task failure, independent of everything else, occurs with probability q₀.
  - Terminal utility: R = V·succ − c·r − L_f·(1 − succ).
- **Value of information.** VOI = Q(h, verify) − Q(h, continue) = p·s·(1 − q₀)·(V + L_f) − c, where p = P(z faulted | h).
- **Group conditions.** Under **A**, all G rollouts of a group share one z; this is ActionProof as trained, where
  `meta.faults` is fixed per task. Under **B**, z is drawn i.i.d. per rollout.
- **Estimators.**
  - GRPO: Â_i = (R_i − R̄_g)/(σ̂_g + ε), with Bessel std and ε = 1e-4 (TRL 1.7.1).
  - Dr. GRPO: R_i − R̄_g.
  - RLOO: R_i − mean_{−i}.
- **Expected update on the verify logit.** U = E[Σ_i Â_i (r_i − ρ)].

## 2. Theorem 1 (finite G, exact)

Assume Condition A and **deterministic within-state returns**: in every hidden state z, R_i = α_z + β_z·r_i (that is,
s ∈ {0, 1} and q₀ = 0). Let k ~ Bin(G, ρ) and define m_G(ρ) = E[√(k(G−k)/(G(G−1))) ; 0 < k < G]. Then, as ε → 0:

  U_GRPO = (G − 1)·m_G(ρ)·Σ_z π₀(z)·sign(β_z),
  U_DrGRPO = (G − 1)·ρ(1 − ρ)·Σ_z π₀(z)·β_z = (G − 1)·ρ(1 − ρ)·VOI,
  U_RLOO = G·ρ(1 − ρ)·VOI.

The Dr. GRPO and RLOO identities need only i.i.d. rollouts given the group's state or states. They hold for any s and
q₀, and under Condition B.

**Proof.**
1. Fix z. Then R_i − R̄ = β_z(r_i − r̄) and σ̂_R = |β_z|·σ̂_r, so Â_i = sign(β_z)(r_i − r̄)/σ̂_r.
2. Σ_i (r_i − r̄)(r_i − ρ) = Σ_i (r_i − r̄)², because Σ_i (r_i − r̄) = 0. This equals (G − 1)σ̂_r². So the group's
   contribution is sign(β_z)·(G − 1)·σ̂_r, and 0 if k ∈ {0, G}.
3. Taking expectations over k gives (G − 1)·m_G(ρ)·sign(β_z). The magnitude |β_z| has cancelled.
4. Without the std, the same steps give β_z·Σ(r_i − r̄)², whose expectation is β_z·(G − 1)ρ(1 − ρ).
5. Averaging over z with π₀ gives the three expressions. Σ_z π₀ β_z = VOI by definition. ∎

**Reading.** GRPO estimates the probability-weighted *majority sign* of the action's effect across hidden states.
Dr. GRPO and RLOO estimate its *expected* effect. For verification there are two state classes:
- outcome states (β > 0): faulted and repairable;
- cost-only states (β = −c): clean, unrepairable, or uninformative.

So GRPO verifies iff P(outcome state) > P(cost-only state), that is, iff p·s > 1/2, whatever c, V and L_f are.
Verified exactly: the threshold is 0.5000 for every c ∈ {0.001, 0.01, 0.05, 0.2} and L_f ∈ {0, 1}
(`test_vote_threshold_is_one_half_and_cost_free`). The Dr. GRPO threshold is exactly c/(s(V + L_f)).

**Generality.** Nothing in the proof is specific to verification. For any binary choice whose payoff is
deterministic given a state shared within the group, GRPO with group std is blind to the stakes. A rare catastrophic
state (small π₀, large |β|) gets one vote of the same size as a common, trivial state. This is the safety-relevant
form of the result. It is stated, not tested beyond verification.

## 3. When the effect weakens or disappears (exact numbers, G = 8, ρ = 0.3)

| Condition | Exact GRPO threshold p* | VOI threshold | Source |
|---|---|---|---|
| A, s = 1, q₀ = 0, c ∈ {0.01, 0.05, 0.2} | 0.495, 0.499, 0.500 | 0.01, 0.05, 0.20 | thresholds |
| A, c = 0.001 (ε = 1e-4 matters: c·σ_r ≈ ε) | 0.452 | 0.001 | thresholds |
| A, s = 0.6 (stochastic recovery) | 0.654, 0.667, 0.706 | 0.017, 0.083, 0.333 | thresholds |
| A, s = 0.3 | never verifies for any p when c ≥ 0.01 | 0.033–0.67 | thresholds |
| A, exogenous q₀ = 0.3 (counterexample) | 0.090, 0.134, 0.281 | 0.014, 0.071, 0.286 | thresholds |
| B (i.i.d. states), s = 1 | 0.233, 0.247, 0.310 | 0.01, 0.05, 0.20 | thresholds |
| Dr. GRPO or RLOO, any condition | = VOI threshold (exact) | | tests |

**Edge cases and counterexamples.**
1. **c = 0.** Cost-only groups have zero variance and contribute nothing. Only outcome groups vote, so GRPO verifies
   whenever p > 0. This is the free-read A12 result.
2. **ε.** For c·σ_r ≲ ε the cancellation is incomplete: the threshold falls to 0.452 at c = 0.001.
3. **Other within-state variance.** In the counterexample, exogenous failure noise in cost-only states dominates
   c·σ_r. The cost-only vote then shrinks to O(c/σ₀), and cost sensitivity returns, biased upward at small c.
   ActionProof's R_V is deterministic given actions and state, so it is in the cost-blind regime.
4. **Stochastic recovery** (s < 1) adds noise inside outcome states and shrinks their vote. The threshold rises
   above 1/2; at s = 0.3 GRPO never verifies.
5. **Condition B.** The scale is still estimated from the same samples. Their composition of states varies; for
   example, an all-clean group has probability (1 − p)^G. The distortion is weaker but present.
6. **ρ ∈ {0, 1}** is absorbing for every estimator: all groups are degenerate.
7. **Multiple decisions per trajectory, token weighting, KL, clipping.** Theorem 1 covers one decision. For the
   multi-decision testbed the prediction is approximate. It is checked empirically below.

**The normaliser condition (v2 §4, restated).** A scale s(·) preserves the expected sign of every decision iff it is
independent of the hidden states of the samples it scales, given the observable context. This is because
E[A/s] = E[1/s]·E[A] when s ⊥ (A, z) | x. A scale computed from the same group, a fork group, a small batch, or a
GDPO per-component group violates the condition. No scale, RLOO, or an EMA of past scales satisfies it.
LAB1-L2N tests this as a causal intervention (H4). It holds the scale's dependence on z fixed while removing the
sample sharing, and the reverse.

## 4. Reproductions from raw files

| Claim | Reproduced value | How |
|---|---|---|
| TB1 wrong-decision rate, GRPO | **319 / 495 = 0.644** | independent recount from `results/ad2/tabular/TB1.jsonl.gz` (`exact_analysis.py: tb1_recount`) |
| TB1 wrong-decision rate, Dr. GRPO | **28 / 495 = 0.057** (rloo and nostd_B identical) | same |
| TB1 VOI equals the closed form p·s·(1 + L_f) − c | max abs deviation 2.2e-16 over 11,664 runs | same |
| Every GRPO error is under-verification | 319 / 319 have VOI > 0 and the policy never verifies; 0 over-verify; mean regret 0.226 | same |
| Every Dr. GRPO error is slow learning, not bias | 28 / 28 have VOI > 0; 27 / 28 have \|VOI\| ≤ 0.05; 26 / 28 at s ≤ 0.6 | same |
| The exact warm-start sign predicts the decision GRPO learns | **494 / 495 cells (0.998)**; condition B 0.976; Dr. GRPO 0.943 | `tb1_predict`: sign of U at ρ₀ = 0.3, ignoring the safe-op read |
| v2 Monte Carlo thresholds | every v2 grid value is the first 0.02 grid point at or above the exact threshold | `mc_check` |
| LLM log decomposition: −143.6 / −0.36; faulted groups +49.6 / −49.6 | **byte-identical** JSON on rerun of `scripts/ad2/decompose.py` on the cluster from `results/ad/rollouts/ad_grpo_s0.jsonl.gz` (sha256 bb0c7a9c…d698) | rerun 2026-11-24 |

**On "+49.6 / −49.6".** Theorem 1 says that each non-degenerate group casts one vote whose size does not depend on
the stakes. It does **not** imply that the faulted-outcome and faulted-cost-only totals are equal. Those totals come
from 20 vs 17 groups with different k. Their near-equality, +49.65 vs −49.59, is a feature of these 160 groups, and
we do not present it as an identity. What the decomposition does show is the predicted sign pattern under std
normalisation:
- clean groups: −143.6;
- faulted groups: a near-cancellation, +0.06;
- centring only: −0.36 net, with faulted groups at +12.2.

## 5. What is and is not claimed

**Claimed: mathematically derived.**
- Theorem 1 and its corollaries, under the stated assumptions.
- The normaliser condition.

**Claimed: reproduced in the testbed.**
- The TB1 rates.
- The 0.998 predictive accuracy of the exact sign.
- The thresholds.

**Claimed: observed in trained agents (descriptive).**
- The LLM training-log decomposition under Condition A: GRPO's group-std arm stopped verifying (first-stage AD
  logs), and the decomposition reproduces the sign pattern.

**Not claimed.**
- That GRPO removes costs in general: see edge case 3.
- That resampling states removes the bias: see edge case 5.
- That the mechanism is new in kind: it is the 1/σ weighting of N1 and the std identity of N4, applied to a hidden
  state.
- Any confirmatory LLM test of the vote rule. V2-E1 (tasks 1300–1303) never ran; §6 of STATE_OF_RESEARCH covers this.
