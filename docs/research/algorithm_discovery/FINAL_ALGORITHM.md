# Algorithm: aliased-world Rao-Blackwellised group advantages (AWRB), and why group-standardised GRPO cannot learn costly verification

Status: the formulation, propositions and tabular checks are final. The LLM results are in §6, filled only from
registered result files.

## 1. Problem

A tool-using agent acts in a POMDP with:
- task x;
- hidden execution world w ~ π₀(· | x), known to the training simulator and unknown to the policy;
- visible history h_t;
- actions a_t ∈ {write, read (verify), repair, escalate, report};
- observations o_t = O(s_t, a_t; w).

**Aliasing.** A write's acknowledgement is identical across worlds (o ⊥ w | h), so after a write the worlds are
aliased until a read.

**Return.** R(τ, w) = R_V(s_T) − c · n_reads(τ), where R_V is the state-verified completion indicator and c ≥ 0 is the
verification cost.

**Objective.** J(θ) = E_{x, w, τ ∼ π_θ}[R(τ, w)].

**Value-of-information rule.** For one aliased decision with fault probability p, repair success probability s and
read cost c:

  read ⟺ p·s > c.

**Training estimator in practice (GRPO).**
- G rollouts per task, all in the task's single world w (in ActionProof, `meta.faults` is fixed per task).
- Â_i = (R_i − μ_g)/(σ_g + ε).

## 2. Propositions

The sign statements below are for one aliased decision with a Bernoulli read policy ρ. They are exact for the
per-world advantage of a two-valued group. The tabular simulation checks them for the full update, with groups of
8 and 4 groups per step:
- `scripts/ad/toy_verify_bandit_v2.py` → `results/ad/t0/toy_bandit_v2.json`;
- `tests/ad/test_ad.py`.

**P1. Group-standardised GRPO makes verification cost-blind (raw returns, group std).**

Within a clean-world group, rewards take two values separated by c. The standardised advantage of a read rollout is
−sqrt((1 − ρ̂)/ρ̂), where ρ̂ is the group's read fraction, for every 0 < c < 1.

Within a faulted-world group with perfect repair, the two values are 1 − c and 0, and the advantage of a read
rollout is +sqrt((1 − ρ̂)/ρ̂).

The expected update on the read logit therefore has the sign of (1 − p)·(−1) + p·(+1), that is sign(2p − 1),
independent of c. With imperfect repair (s < 1) the faulted-world term shrinks, and the threshold moves above 1/2.

At c = 0 exactly, clean groups have zero variance and contribute nothing, so GRPO reads whenever any faulted group
rewards reading. That is why A12 learned read-back 1.00. Any c > 0, however small, switches the clean groups from
silent to full-strength votes against reading.

This is the mechanism of Bereket and Leskovec (arXiv 2508.11800): group std produces policy-dependent weights 1/σ_x
over a group-shared outcome. What is new here is its consequence for information acquisition.

Toy check:
- reads 0.001 at p = 0.45, c = 0.05 (optimum: read);
- reads 0.001 at p = 0.6, s = 0.6 (optimum: read).

**P2. Without std scaling the sign is the VoI sign (raw/none, i.e. Dr. GRPO's std removal).**

The expected per-world advantage is a reward difference, so the update is proportional to

  (1 − p)(−c)(1 − ρ) + p(s − c)(1 − ρ) ∝ p·s − c.

This is unbiased in expectation, up to GRPO's group-mean leave-in factor (G − 1)/G (Bereket and Leskovec).

**P3. The Rao-Blackwellised return (AWRB).** Define

  R̄(τ) = E[R | τ, x] = Σ_{w : O(τ; w) = o(τ)} π₀(w) R(τ, w) / Σ_{w : O(τ; w) = o(τ)} π₀(w).

Because π_θ depends on w only through observations, every world in the sum gives τ the same likelihood. Then:

- (a) E[R̄] = E[R]; Var(R̄) ≤ Var(R); and E[∇ log p_θ(τ) R̄] = ∇J.
- (b) A trajectory that never observes the world gets R̄ = the prior mean. A trajectory that reads keeps its
  revealed-world return.
- (c) R̄ is computable exactly by deterministic replay of τ's calls in every prior world, with zero extra generation.

(a) and (c) are tested exactly by enumeration. Two tests, `test_marginal_matches_independent_world_enumeration` and
`test_unbiased_and_rao_blackwell`, run the scripted policies afresh in every world; they do not replay.

**P4. AWRB with group std has the mirror bias (rb/group).**

In a clean-world group, a read rollout has R̄ = 1 − c and a no-read rollout has R̄ = 1 − p, so reading wins iff p > c.

In a faulted-world group, a read rollout has R̄ = 1 − c with probability s and −c otherwise; no-read is still 1 − p.

Std scaling gives each group an O(1) vote. When p < 1/2 the clean groups dominate, and the rule becomes "read iff
p > c", whatever s is. AWRB with group std therefore over-verifies when the agent cannot repair (s small). It reads
at s = 0 in the toy, where reading is worthless.

**P5. AWRB without std (rb/none) has the VoI sign** (as P2), and lower or equal return variance (P3a).

Its gradient SNR is higher than raw/none's when repair succeeds:

| Setting | rb/none SNR | raw/none SNR |
|---|---|---|
| s = 1 | 10.7 | 1.08 |
| s = 0.6 | 1.09 | 0.84 |
| p = 0.15 | 0.45 | 0.33 |

It is lower when repair never succeeds: |SNR| 0.22 vs 10.7 at s = 0.

The reason is that the raw within-group comparison is already paired by the shared world, a common-random-number
effect. Replacing no-read returns by the prior mean breaks that pairing. Rao-Blackwellising the *return* does not
guarantee lower variance of a *baseline-subtracted, group-paired* advantage. The same family of caveat appears in
Sakib (arXiv 2609.24144) and in mandate Prop. 3.

**Summary of decision rules (one aliased decision, p < 1/2):**

| Estimator | Learned rule | Correct? |
|---|---|---|
| GRPO (raw, group std) | read iff p > 1/2 (s = 1), stricter for s < 1; independent of c | no |
| AWRB with group std | read iff p > c; independent of s | no (over-verifies when repair fails) |
| raw, no std (Dr. GRPO advantage) | read iff p·s > c | yes; noisier |
| AWRB, no std (**awrbn**) | read iff p·s > c | yes; less noisy when repair works |

## 3. Algorithm (AWRB-n)

```
Input: tasks x with a known world prior pi0(w|x); policy pi_theta; cost c; group size G.
repeat for each training step:
  for each prompt x (world w* drawn by the pool, hidden from the policy):
     sample G rollouts tau_i ~ pi_theta in env(x, w*)          # unchanged GRPO sampling
     for each tau_i:
        C_i = { w in supp pi0(.|x) : replaying tau_i's calls in env(x, w) reproduces every observation }
        Rbar_i = sum_{w in C_i} pi0(w) R_V(replay final state) / sum_{w in C_i} pi0(w)  -  c * reads(tau_i)
     A_i = Rbar_i - mean_j Rbar_j                              # no std division
  GRPO/DAPO clipped token loss with A_i on policy tokens, KL to the reference
```

Implementation: `actionproof/ad/worlds.py` (`CWEnv`, `world_prior`, `replay`, `marginalise`),
`actionproof/ad/rewards.py` (`awrbn`), `actionproof/ad/train.py`.

**Complexity.**
- Per trajectory, O(|supp π₀| × T) environment calls on the CPU. That is 8–14 worlds × ≤ 16 calls here, about 1 s
  per training step against roughly 25 s of generation.
- No extra rollouts.
- No extra forward or backward passes.

## 4. Assumptions and limitations

1. **The world prior is known at training time.** This is a simulator assumption; deployment needs nothing.
   Misspecification biases R̄ toward the assumed prior. Not yet ablated.
2. **The environment is deterministic given (x, w, actions, id seed).** Stochastic tool latency or ordering would
   make observation-equivalence probabilistic. The posterior weights would then need the likelihood, not an
   indicator.
3. **Support size.** |supp π₀| must be small enough to enumerate. With many independent faults per episode, the
   product space needs sequential pruning or sampling.
4. **No theoretical guarantee about learning speed is claimed.** Only unbiasedness (P2, P3a, P5) and the sign
   statements (P1, P4) for one decision are claimed.
5. **Novelty.**
   - The group-std mechanism is Bereket and Leskovec's.
   - Std removal is Dr. GRPO's.
   - Rao-Blackwellisation over exogenous inputs is classic (Mao et al.; SAMPLR is the closest).
   - What is ours: the consequences P1 and P4 for verification decisions, the observation-equivalent replay
     estimator for LLM agents, and the LLM-agent tests.

## 5. Why this is a change to the optimisation procedure, not a reward

Every arm optimises the same J(θ) with the same reward definition, pool, rollouts and steps. AWRB changes the return
estimator, a conditional expectation with the same mean. The no-std arms change the advantage normalisation. Neither
changes what the agent is rewarded for.

## 6. LLM results

### 6.1 P1 in the LLM agent (development data, descriptive)

**Collapse.** grpo with c = 0.05 on cw_train drives training-time reads from 0.59 to 0.03 per episode in seed 0, and
from 0.56 to 0.07 within 25 steps in seed 1. Zero-variance groups reach 86%. Greedy cw_dev behaviour ends exactly at
the warm start's: 0 reads, VSC 0.707, in both seeds (`results/ad/dev/ad_grpo_s{0,1}__cw_dev.summary.json`).

**Mechanism.** We recomputed the advantages of grpo s0's logged training returns, steps 0–40, under both
normalisations (`results/ad/advantage_decomposition_steps0-40.json`). With centring only, the summed advantage of
read rollouts is −12.5 from clean-world groups and +12.2 from faulted-world groups: net −0.4, about the true value of
information at the warm start's repair ability. With group std it is −143.6 and +0.06. Clean groups, whose only
spread is the read cost, are amplified by 1/σ_w and dominate. This is the 1/σ_w world weighting of P1, observed
directly.

### 6.2 Gates and confirmatory results

PENDING: filled from `results/ad/d1_summary.json`, `d2_summary.json`, `d3_summary.json` and `c1_summary.json` only.
