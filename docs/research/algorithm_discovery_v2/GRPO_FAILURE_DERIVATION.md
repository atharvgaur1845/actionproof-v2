# Why group-standardised GRPO stops verifying when reads cost something

Status: derivation and numerical demonstration are final. The tabular sweep evidence is in
`CONTROLLED_TESTBED.md`; the LLM training-log decomposition is reproduced in §6.

**Scope of the claim.** We do **not** claim that GRPO removes action costs in general. We claim a precise
distortion under stated conditions (§5). The group-std mechanism itself is Bereket and Leskovec's
(arXiv 2508.11800). Removing the std is Dr. GRPO's (Liu et al., arXiv 2503.20783). Our part is:
- the consequence for a sequential agent's information-gathering action;
- the exact conditions under which it occurs;
- the "vote" reading of it;
- how it extends to batch, fork-group and per-component (GDPO) normalisers.

## 1. Model

ActionProof as a POMDP:
- **Task and hidden state.** A task x has a hidden execution state z ~ π₀(·|x): the fault plan (which writes
  silently failed, partially applied, were duplicated or timed out). The policy never observes z.
- **Histories.** h_t = (x, a_1, o_1, …, a_{t−1}, o_{t−1}). The policy is π_θ(a | h_t). Observations satisfy
  o_t = O(s_t, a_t; z). A write's "ok" acknowledgement is the same in every z: **aliasing**.
- **Belief.** b_t(z) = P(z | h_t). For one write w: p_f(h_t) = b_t(write w silently failed).
- **Read decision.** After an ok acknowledgement the agent chooses verify (cost c_v) or skip. If it verifies and
  the read shows a fault, a recovery succeeds with probability q_r(h_t). Recovery avoids the loss L_f(h_t): the
  task value V plus any failure penalty.
- **Return.** R(τ, z) = V·succ − c_v·n_reads − L_pen·(1 − succ) − D·n_dup.

**Single-decision value of information (myopic, perfect read):**

  VOI(h_t) = p_f(h_t) q_r(h_t) L_f(h_t) − c_v(h_t).

**Extensions** (all computed exactly by the testbed solver, `actionproof/ad2/tabular/env.py:root_voi`):
- **Several fault classes** k with probabilities p_k(h) and recovery rates q_k:
  VOI = Σ_k p_k q_k L_k − c_v. Classes with q_k ≈ 0 (e.g. partial payments the LLM cannot repair) add nothing.
- **False alarms.** A clean write raises a fault flag with rate f, and "repairing" a clean write duplicates it
  (loss D): VOI = p q L − (1 − p) f D·1[act on flag] − c_v. When acting on a flag is not worth its expected
  duplicate cost, the posterior after a flag decides.
- **Irreversible and duplicate-prone actions.** After a visible timeout the choice is three-way: proceed, blind
  retry, or verify. Retry duplicates an applied write with probability a, at loss D. Verifying is worth
  min{(1 − a)L, aD} − c_v over the better blind action.
- **Multiple simultaneous faults.** If the task succeeds only when every write is correct, then
  L_f(h_t) = V·P(all other writes end correct | h_t, continuation). The VOI of one read depends on the other
  decisions: it is non-myopic, and it is computed by expectimax over the history tree.
- **Stopping.** A re-read of a record already read, with no write in between, has p_f = 0 under the posterior,
  so VOI = −c_v. "Stop verifying once sufficient evidence exists" is the statement that VOI(h) < 0 as soon as
  b_t has collapsed.

## 2. What GRPO estimates

For a group of G rollouts τ_1, …, τ_G of task x, TRL 1.7.1 with `scale_rewards="group"` computes

  Â_i = (R_i − R̄_g) / (σ̂_g + ε),

where σ̂_g is the Bessel-corrected std (`trl/trainer/utils.py:nanstd`) and ε = 1e-4. The update is
Σ_i Â_i ∇log π_θ(τ_i), with a token-level sum ('dapo' loss).

**Condition A (ActionProof as trained).** All G rollouts of a group share one hidden state z_g. In the LLM,
`meta.faults` is fixed per task.

**Lemma 1 (per-group affine invariance).** For a > 0 and any b, with ε → 0, Â is unchanged by R ↦ aR + b inside
one group. A group's update therefore depends only on the *pattern* of its returns, never on their scale.

**Proposition 1 (GRPO optimises a world-reweighted mixture).** As G grows, σ̂_g → σ_{x,z}(θ) = sd(R | x, z),
and

  E[ĝ] ≈ ((G − 1)/G) Σ_x Σ_z π₀(z|x) ∇_θ J_{x,z}(θ) / σ_{x,z}(θ),  with J_{x,z} = E[R | x, z].

The true gradient is Σ_x Σ_z π₀(z|x) ∇J_{x,z}. GRPO weights each *hidden state* by 1/σ_{x,z}(θ). The weight
depends on z (which the policy cannot see) and on θ. For finite G the approximation has an extra correlation
term between σ̂ and the centred returns. That term is the subject of §4.

## 3. The verification decision under Condition A

Take one aliased verify-or-skip decision, read with probability ρ. Classify each hidden state by what verifying
changes in that state.

**Cost-only states C.** Verification cannot change the outcome. Examples:
- a clean write;
- a fault the agent cannot repair (q_r = 0);
- a read that is uninformative given the history.

Then R = R_0 − c·r with r ∈ {0, 1}. The within-group std is c·sqrt(ρ(1−ρ)), and ∂J_z/∂ρ = −c. After
standardisation the update on ρ is

  −c / (c·sqrt(ρ(1−ρ))) = −1/sqrt(ρ(1−ρ)),  **independent of c**.

**Outcome states O.** Verification changes the outcome. In the deterministic-recovery case, R(read) = R_0 + Δ − c
and R(skip) = R_0, with Δ = L_f. The std is (Δ − c)·sqrt(ρ(1−ρ)) and ∂J_z/∂ρ = Δ − c. Standardised:
+1/sqrt(ρ(1−ρ)), **independent of Δ and c**.

**Proposition 2 (the update is a vote).** Under Condition A with deterministic recovery, GRPO's expected update on
the read logit is

  ∝ sqrt(ρ(1−ρ)) · [P(O) − P(C)] = sqrt(ρ(1−ρ)) · (2·p_f·q_r − 1).

Each non-degenerate group casts one O(1) vote for the action taken by its better rollouts. GRPO learns to verify
iff **p_f·q_r > 1/2**, whatever c, V or L_f.

The value-of-information rule is p_f·q_r·L_f > c. The two rules agree only by accident.

**Corollaries.**
- **(a)** With stochastic recovery, outcome groups also contain failed reads. Their vote shrinks, and the
  threshold rises above 1/2.
- **(b)** At **c = 0**, cost-only groups have zero variance. TRL drops them from the gradient (zero advantage).
  Only outcome groups vote, so GRPO verifies whenever P(O) > 0. This is the first-stage A12 result: free reads,
  read-back 1.00.
- **(c)** The **mirror bias** of first-stage AWRB with group std (`awrb`). Replacing a no-read return by its
  prior mean changes the per-state patterns:
  - in a clean group, a read yields 1 − c, and a skip yields 1 − p(1 + L_pen);
  - so the clean-group vote is "read iff p(1 + L_pen) > c", whatever q_r is.

  awrb over-verifies when recovery fails (FINAL_ALGORITHM P4).
- **(d)** A **Lagrangian cost** λ·c·n_reads (constrained GRPO) is invisible to the clean-group vote: λ cancels by
  Lemma 1.
- **(e)** **GDPO** (`multi_objective_aggregation="normalize_then_sum"`, arXiv 2601.05242) standardises each reward
  component inside the group. The cost component becomes −(n − n̄)/sd(n): an O(1) anti-read vote in *every*
  group where reads vary, faulted groups included. That outweighs the utility vote unless almost every group is
  an outcome group, so GRPO-with-GDPO stops verifying for any c > 0 (§7).

## 4. Normaliser independence: the general condition

Write any scaled estimator as ĝ = Σ_i (A_i/s) ∇log π(τ_i), where A_i is an unbiased centred advantage and s > 0
is a scale.

If s is statistically independent of the hidden states and returns of the samples it scales, given the
observable context, then E[ĝ | x] = E[1/s | x]·E[Σ_i A_i ∇log π]. Each task's gradient is then rescaled by a
positive constant, and **the sign of every decision is preserved**.

If s is computed from samples whose hidden-state composition varies, then 1/s correlates with A, and the
expectation is a hidden-state-reweighted mixture: Proposition 1, with the mixture taken over compositions.

This single condition covers every case:

| Normaliser | Computed from | Correlated with the hidden state? | Prediction |
|---|---|---|---|
| group std, Condition A | one group = one hidden state | yes, fully | vote rule p·q > 1/2 (§3) |
| group std, Condition B (z i.i.d. per rollout) | a mixture of ≈ G draws | yes, through the composition: an all-clean group has probability (1 − p)^G | bias shrinks with G·p; not removed for p ≲ 1/G |
| batch std (TRL "batch"), 4 groups | the step's 4 hidden states | yes, at batch level: P(all-clean batch) = (1 − p)^4 | biased, less than group std |
| fork-group std at a decision | K forks' sampled worlds | yes, P(all-clean forks) = (1 − p)^K | biased (TB2 `fork_bel_std`) |
| EMA of past std (first-stage D2), per-context EMA | earlier steps only | no | sign-preserving |
| none (Dr. GRPO), RLOO | — | — | sign-preserving (P2) |

The condition-B row corrects the first-stage protocol's informal claim, and the hypothesis H1c that we registered
in V2TB_PROTOCOL §4. Resampling the world per rollout does **not** fully remove the distortion at finite G. The
numerical demonstration (§7) shows a c-dependent but biased threshold under Condition B. The bias falls only as
(1 − p)^G → 0.

## 5. When the distortion occurs, and when it does not

GRPO's verification threshold is cost-blind when **all** of the following hold:

1. **Scale from the same samples.** The group scale is computed from samples that share (or are composed of a
   small number of) hidden states: Condition A, or small G·p under Condition B.
2. **Cost is the only variance in some states.** There are hidden states in which verification changes only the
   cost, and in which the cost is the main source of within-group reward variance:
   σ_C ≈ c·sqrt(ρ(1−ρ)) ≫ any other variance.
3. **The cost is above ε.** 0 < c and c·sqrt(ρ(1−ρ)) ≫ ε = 1e-4. At c ≈ 1e-3 the ε term restores some scale
   dependence (`tests/ad2/test_tabular.py::test_grpo_clean_group_advantage_is_cost_free`).
4. **The policy is non-degenerate.** 0 < ρ < 1. At ρ = 0 every group is degenerate; the state is absorbing for
   every estimator.

**Counterexample (GRPO does not remove costs).** Violate condition 2: give every task an exogenous failure
probability q_0 = 0.3, independent of verification. Clean groups are then no longer cost-only; their std is
dominated by sqrt(q_0(1 − q_0)) ≫ c. Their vote shrinks to O(c/σ_0), and GRPO's threshold becomes c-dependent
again. Measured thresholds:

| c | GRPO threshold | VOI threshold |
|---|---|---|
| 0.01 | 0.10 | 0.014 |
| 0.05 | 0.14 | 0.071 |
| 0.20 | 0.28 | 0.286 |

The threshold is still biased upward at small c. How badly GRPO distorts a cost therefore depends on the ratio
of the cost-induced variance to all other within-group variance in the states where verification does not matter.
In ActionProof that ratio is large: R_V is deterministic given the actions and the hidden state, so clean-world
groups vary *only* through reads (§6).

## 6. The five candidate explanations, separated

| # | Explanation | Verdict | Evidence |
|---|---|---|---|
| 1 | Group-wise scale invariance | **Necessary.** It is the operation that removes c (Lemma 1). Not sufficient alone: with i.i.d. worlds and large G·p the distortion vanishes. | Proposition 2; derivation_numeric (Condition B large p) |
| 2 | Over-weighting of low-variance clean groups | **The mechanism, as a consequence of 1 and 3.** More precisely, every cost-only group is over-weighted, clean or not. In the LLM, *faulted* groups whose reads did not lead to a repair (`fault_cost`) pull against reading as hard as successful-repair groups pull for it. | §6 table: fault_cost −49.6 vs fault_outcome +49.6 |
| 3 | Conditioning on a fixed latent fault inside each group | **Necessary in its general form:** the scale must correlate with the hidden state. Condition A is the extreme case. Condition B weakens it but does not remove it at finite G. | §4; derivation_numeric |
| 4 | Interaction with the action distribution and warm start | **Not the cause of the sign.** The vote's sign is independent of ρ (threshold 0.52 for ρ0 ∈ {0.05, 0.3, 0.6, 0.9}). ρ sets the speed, ∝ sqrt(ρ(1−ρ)). ρ = 0 is absorbing for every estimator: the first-stage grpo reached 86% zero-std groups. The warm start's *recovery skill* q_r sets how many faulted groups are cost-only, and so shifts the threshold. | derivation_numeric `rho_dependence`; TB6 |
| 5 | Reward design / implementation unrelated to normalisation | **Excluded as the cause of grpo vs nostd.** The two arms differ only in `scale_rewards`: same reward, pool, loss, LoRA and seeds. Implementation details change magnitudes only. | §6 below |

Details for row 5:
- the 'dapo' token-sum weights longer read rollouts more. The token-weighted net is −811.6 under std and +15.9
  under centring only;
- every first-stage rollout has env and id seed 0, an audit finding (AWRB_IMPLEMENTATION_AUDIT §3). This changes
  no reward.

## 7. Numerical demonstration (`scripts/ad2/derivation_numeric.py` → `results/ad2/derivation_numeric.json`)

**Method.** The expected update on the read logit, E[Σ_i Â_i (r_i − ρ)] per group, over 40,000 groups of G = 8
at ρ = 0.3. For each setting, the smallest p on a 0.02 grid at which the update is positive:

| Estimator, condition | s | c = 0.01 | c = 0.05 | c = 0.2 | VOI threshold c/s (L_pen = 0) |
|---|---|---|---|---|---|
| grpo, A | 1.0 | 0.50 | 0.52 | 0.52 | 0.01 / 0.05 / 0.20 |
| grpo, A | 0.6 | 0.66 | 0.68 | 0.72 | 0.017 / 0.083 / 0.333 |
| grpo, A, L_pen = 1 | 1.0 | 0.50 | 0.52 | 0.52 | 0.005 / 0.025 / 0.10 |
| grpo, B | 1.0 | 0.24 | 0.26 | 0.32 | 0.01 / 0.05 / 0.20 |
| grpo, A, exogenous q0 = 0.3 | 1.0 | 0.10 | 0.14 | 0.28 | 0.014 / 0.071 / 0.286 |
| nostd (Dr. GRPO), A | 1.0 | ≤ 0.02 | 0.06 | 0.20 | 0.01 / 0.05 / 0.20 |
| nostd, A | 0.6 | ≤ 0.02 | 0.10 | 0.34 | 0.017 / 0.083 / 0.333 |
| gdpo, A | 1.0 | 0.98 | never | never | 0.01 / 0.05 / 0.20 |
| any, c = 0 | — | — | — | — | GRPO threshold ≤ 0.02 (always verify) |

**What the table shows.**
- GRPO's threshold is the vote threshold 1/(2s) (≈ 0.5 at s = 1, ≈ 0.7 at s = 0.6), flat in c and in L_pen.
- Dr. GRPO's threshold tracks the VOI threshold.
- GDPO is cost-blind in the opposite direction: it never verifies.
- Condition B is cost-sensitive but biased.

## 8. Reproduction of the LLM training-log decomposition (−0.4 vs −143.6)

**Command.**
`python scripts/ad2/decompose.py --tags ad_grpo_s0 ad_grpo_s1 ad_nostd_s0 ad_awrb_s0 --steps 0 40 --out results/ad2/decomposition.json`

**Inputs.** This is an independent re-implementation, not the first-stage script.
- **Source:** `results/ad/rollouts/ad_grpo_s0.jsonl.gz`, the first-stage training log; one record per rollout.
- **Steps:** 0–39, i.e. 40 steps × 4 tasks = 160 groups of 8 rollouts each.
- **Reward:** R = R_V − 0.05·n_verify. n_verify counts every agent read-tool call, including PO and journal reads.
- **World class:** the group's pool world (`fault_class`).
- **Aggregation:** the sum of advantages over rollouts with n_verify > 0, by world class.

| Denominator | Clean groups | Faulted, outcome varies | Faulted, cost-only | Net |
|---|---|---|---|---|
| pstdev + 1e-4 (first-stage script) | **−143.63** | +49.65 | −49.59 | **−143.57** |
| Bessel std + 1e-4 (TRL 1.7.1 exact) | −134.37 | +46.44 | −46.39 | −134.32 |
| none (centring only) | **−12.54** | +14.92 | −2.73 | **−0.36** |

**Group counts.** 53 clean groups, 20 faulted-outcome groups and 17 faulted-cost-only groups vary; 70 groups
have zero variance.

**What this identifies.**
- **The reported numbers are reproduced exactly:** −143.6 (std) and −0.36 (centring).
- **Denominator.** The first-stage −143.6 uses the *population* std. TRL's actual denominator gives −134.3. The
  conclusion is unchanged.
- **Groups.** The faulted-world near-zero under std (+0.06) is not "no signal". It is a cancellation:
  outcome-changing faulted groups push for reading, and faulted groups whose reads led to no repair push against
  it equally hard. That is the vote rule of §3, observed directly.
- **Seed 1:** −139.6 clean, net −119.3 (std) vs net −0.19 (centring).
- **nostd s0's own rollouts:** net −215.8 under std vs −2.5 under centring.

**Caveat.** These are sums of sequence advantages over read rollouts, a proxy for the read-logit update. They are
not the gradient itself, which depends on token counts and score functions. Weighting each rollout by its call
count (a token proxy), the std net is −811.6 and the centring net is +15.9 (s0).

## 9. Summary of conditions (the defensible statement)

> Under group-relative advantage normalisation, a verification action's cost cancels inside every rollout group
> in which verification changes only cost. This happens when the group's scale is estimated from samples that
> share, or are dominated by, one hidden execution state, and when the cost is the dominant within-group reward
> variance in such states. The policy then verifies iff the probability that verification changes the outcome
> exceeds about 1/2. The value-of-information threshold p·q·L > c plays no role.
>
> Scale estimates that are independent of the current samples' hidden states preserve the VOI sign. Examples: no
> scale, a leave-one-out baseline, or an EMA of past scales.

**What is not claimed.**
- That GRPO always removes costs: see the §5 counterexample.
- That resampling worlds fully fixes the problem: see §4.
- That the mechanism is new in kind: it is Bereket and Leskovec's 1/σ weighting with the hidden state in place of
  the stochastic outcome.
