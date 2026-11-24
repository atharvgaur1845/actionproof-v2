# Group-normalised policy gradients when rollouts share a hidden state: statements, proofs, assumptions

Branch `research/latent-state-theory`. Reconstructed from the code (`actionproof/ad/*.py`, TRL 1.7.1
`GRPOTrainer._generate_and_score_completions`, `actionproof/ad2/tabular/*`), not from earlier reports.
Every identity below is checked exactly by enumeration in `tests/theory/test_exact_theory.py` (44 tests, including
a cross-check against the independently written `actionproof/lab/exact_vote.py`). Numbers marked [scan] come from
`results/theory/scan.json` (`scripts/theory/scan.py`, Slurm).

Evidence label for this file: **[math]** (derived; exact-enumeration checks).

---

## 0. What the implementation actually does (the facts the theory must match)

| Question | Answer from source | Where |
|---|---|---|
| Is the hidden fault state shared within a GRPO group? | **Yes, by construction.** `make_pool` draws one fault plan per task (`t.meta["faults"]`), and `make_env_cw(task, rid, seed)` reads that plan for every rollout of the task. TRL groups the `num_generations = 8` completions of one prompt; prompts are unique per task (`assert len(by_prompt) == len(tasks)`). So every group is one task in one fault world. | `actionproof/ad/pools.py` L26–32; `actionproof/ad/worlds.py` `make_env_cw`; `actionproof/ad/rollout.py` L22–24; `actionproof/ad/train.py` `num_generations=8` |
| Is there environment randomness inside a group? | **Only ids.** `ProcEnv.rng` is used only by `_next_id`, and `CWEnv` replaces ids with a hash of (rollout seed, call, draw). Given the task and the action sequence, the final state and R_V are deterministic. All within-group variation comes from the policy's sampling. | `actionproof/integrity/sim/env.py` L65, L140; `worlds.py` `CWEnv._next_id` |
| Reward | R = R_V − c·n_verify, R_V ∈ {0, 1} (all requested steps verified in the final DB state), c = 0.05 (0.30 in the c30 interventions); `awrb*` arms use E[R_V \| trajectory] under the cw1 prior. | `actionproof/ad/rewards.py` |
| Normalisation | `scale_rewards="group"`: A = (R − mean_g)/(std_g + 1e-4), with Bessel std (`trl.trainer.utils.nanstd`, correction count/(count−1)). `"none"` (the `nostd` arm) centres only. | TRL 1.7.1 `grpo_trainer.py` L2338–2358, `utils.py` L752–775 |
| Loss aggregation | `loss_type` is not set, so TRL's default **`"dapo"`**: token losses are summed and divided by the number of active tokens in the accumulated batch. One optimizer step per 32 rollouts (4 prompts × 8, `gradient_accumulation_steps=32`), `num_iterations=1`, so the importance ratio is 1 and clipping is inactive (up to vLLM/trainer log-prob mismatch). KL β = 0.02 to the warm start. | `actionproof/ad/train.py`; TRL `grpo_config.py` L244, L760 |
| Consequence for naming | The `nostd` arm is "GRPO without the std" with DAPO token aggregation. Dr. GRPO (Liu et al. 2025) also replaces per-sequence length normalisation by a constant. Under `dapo` aggregation the two differ by one positive batch-level scalar (active-token count vs. a constant), which changes step size but not the direction within a step. We call the arm "Dr. GRPO (std removed)" and state this once. | — |
| Multi-step structure | A trajectory contains several decisions (which reads, retries, repairs). Repair success after a read depends on the policy's later sampled actions. So in the LLM, "s" is **rollout randomness of the policy itself**, not a property of the state. | `actionproof/ad/rollout.py` |
| The tabular testbed TB1 | Same structure: one shared world per group (cond A), with `ro_j ~ Bern(s)` (repair success) and `fa_j ~ Bern(f)` resampled per rollout (rollout noise). | `actionproof/ad2/tabular/env.py` docstring; `train.py` L219–222 |

**Correction to the mission statement.** The "494 of 495" figure (`results/lab/exact_analysis.json`,
`tb1_predict`) counts **runs** (cells × 3 seeds), not cells. It uses the exact enumerated sign of the warm-start
update at π₀ = 0.3, with s treated as rollout noise. It does **not** test the rule p·s > 1/2. At cell level, the plain
rule p·s > 1/2 agrees with GRPO's learned decision in 149/159 TB1 cells (0.937), and the VOI rule in 59/159 (0.371). For
Dr. GRPO, the VOI rule agrees in 155/159 (0.975) and p·s > 1/2 in 53/159 (0.333). (Recount: cells with c ≥ 0.01 and
|p·s·(1+L) − c| > 0.02, mean over 3 seeds; `docs/theory/THEOREM_AUDIT.md` §2.)

---

## 1. Setting and notation

One binary decision, then a fixed downstream rule. This is the verification decision after an "ok" acknowledgement.

- **Hidden state.** z ∈ 𝒵 (finite), prior w(z). In the canonical verification problem 𝒵 = {clean, faulted}, with
  w(faulted) = **p** = P(the acknowledged write was silently not applied | the agent's history). In the code this is
  the posterior fault mass of the cw1 prior at that site; in TB1 it is the op type's `p`.
- **Action.** r ∈ {0 skip, 1 verify}, π = π_θ(r = 1 | h) = σ(θ). Assumption **A1 (aliasing):** r is drawn
  independently of z. The policy sees only h, and every z gives the same h.
- **Observation and outcome.** After r the rollout receives an outcome o ~ P(o | z, r): read result, whether the
  repair succeeded, exogenous failure. The return is R = R(z, r, o).
  - **s** = P(repair succeeds | fault detected). It is either a *state* property (the fault is repairable or not;
    "split" below) or *rollout* randomness ("noise").
  - **d** = detection probability; **f** = false-alarm probability; **c** = read cost (possibly state-dependent);
    **V** = value of success; **L** = loss of an unrepaired fault; **D** = duplicate penalty; **q₀** = exogenous
    failure.
  - Canonical problem: R = V·succ − L·(1 − succ) − c·r.
- **Group.** G rollouts. A *group law* Q is a joint law of (z₁, …, z_G). Assumption **A3:** Q is exchangeable with
  marginal w. Assumption **A2:** given the states, the rollouts are independent.
- **Laws used.**
  - *shared:* z₁ = … = z_G ~ w. This is ActionProof as trained.
  - *iid:* z_i ~ w independently.
  - *mix(ρ):* shared with probability ρ, else iid.
  - *dirmult(ρ):* z_i | θ ~ θ i.i.d. with θ ~ Dirichlet(((1−ρ)/ρ)·w).
  - Both families have pairwise intra-class correlation ρ for every state indicator (`pairwise_icc`, tested).
- **Estimators.** Centred advantage d_i = R_i − R̄, with R̄ = (1/G)Σ R_j.
  - GRPO: A_i = d_i/(σ̂ + ε), σ̂² = Σ d_j²/(G − 1), ε = 1e-4.
  - Dr. GRPO (std removed): A_i = d_i.
  - RLOO: A_i = G/(G−1)·d_i.
- **Expected update** on the logit: U(π) = E[Σ_i A_i (r_i − π)], since ∂ log π(r)/∂θ = r − π.
- **Within-state effect and its average.**
  - β_z = E[R | z, r = 1] − E[R | z, r = 0].
  - ΔQ = Σ_z w(z) β_z = Q(h, verify) − Q(h, skip) under the fixed downstream rule.
- **Vote.** 𝒱 = Σ_z w(z)·sign(β_z).

**VOI.** Let gross VOI(h) = E_o[max_a Q(h, o, a)] − max_a E_o[Q(h, o, a)] be the decision-theoretic value of the read's
observation, and net VOI = gross VOI − E[c]. Then **ΔQ = net VOI iff the downstream rule after each observation
(repair on an alarm, otherwise continue) is Bayes-optimal**, and no action available without reading would have used
the information. In the canonical family (no blind repair without an observation) this holds whenever repairing on
an alarm is worthwhile: V + L > 0 for faults and the false-alarm duplicate penalty is not too large. With a false
alarm f > 0 and a duplicate penalty D, ΔQ < net VOI if "repair on alarm" is not optimal for that (f, D). In a policy
whose downstream behaviour is itself being learnt, ΔQ is Q^π(h, 1) − Q^π(h, 0): the on-policy advantage, not VOI.

---

## 2. Results that hold for every group law

### Proposition 2.1 (the group update does not depend on π; Bernstein form)
Under A1–A3, for any centred estimator (Σ_i A_i = 0 in every group):

  U(π) = Σ_{k=0}^{G} C(G,k) π^k (1−π)^{G−k} b_k,   b_k = E[Σ_{i: r_i=1} A_i | exactly k readers],

with b₀ = b_G = 0, and every b_k independent of π.

*Proof.*
1. Σ_i A_i (r_i − π) = Σ_i A_i r_i − π Σ_i A_i = Σ_{readers} A_i.
2. Condition on the set S of readers, |S| = k. By A1 the states are independent of S, so the joint law of
   (z, o, R) given S is the same for every π. By exchangeability (A3) it depends on S only through k.
3. P(|S| = k) = C(G,k)π^k(1−π)^{G−k}. ∎

**Consequences.**
- (i) U′(0) = G·b₁ and U′(1) = −G·b_{G−1}. The never-verify corner is locally stable iff b₁ < 0; the always-verify
  corner is locally stable iff b_{G−1} > 0.
- (ii) Write U = (1−π)^G Σ_k C(G,k) b_k t^k with t = π/(1−π) ∈ (0, ∞). By Descartes' rule of signs, the number of
  interior zeros of U is at most the number of sign changes in (b₁, …, b_{G−1}). With no sign change the flow ends at a
  corner; one sign change allows one interior zero, and so on.
- (iii) The 1-D mean-field flow dθ/dt = U(σ(θ)) from π₀ moves in the direction of sign U(π₀) and stops at the first
  zero of U in that direction (`exact.endpoint`).

### Proposition 2.2 (centred estimators are invariant to the group law)
Under A1–A3, for Dr. GRPO, b_k = k(G − k)/G · ΔQ, so

  U_Dr(π) = (G − 1) π(1 − π) ΔQ  and  U_RLOO = G π(1 − π) ΔQ,

**for every exchangeable group law**: shared, iid, mixtures, Dirichlet-multinomial.

*Proof.*
- Σ_{readers} d_i = (1 − k/G) Σ_{readers} R_i − (k/G) Σ_{skippers} R_j.
- Given k, each reader's state has marginal w (A1, A3), so E[R_i | reader] = μ₁ := Σ_z w(z) E[R | z, 1]. Likewise
  μ₀ for skippers.
- Hence b_k = (1 − k/G) k μ₁ − (k/G)(G − k) μ₀ = k(G − k)/G · (μ₁ − μ₀).
- Summing with Bernstein weights uses E[k(G − k)] = G(G − 1)π(1 − π). ∎

**General form (multi-step).** For any policy, E[Σ_i (R_i − R̄) ∇log π_θ(τ_i)] = (G − 1)·∇J(θ) for any exchangeable
law with conditionally independent rollouts. Cross terms vanish because E[∇log π_θ(τ_i) | z₁..z_G] = 0: the dynamics
given the state do not depend on θ. This is the policy-gradient theorem with a leave-in baseline. It is **not new**; the
only point added here is that it is unaffected by how strongly the group's states are correlated. The correlation
changes only the variance.

**Assumptions it needs, and what breaks it.**
- **A1.** If the decision depends on information correlated with z (e.g. an earlier read), the result holds per
  observable history; the multi-step form still holds.
- **A2.** Correlated decisions across rollouts, e.g. a shared token prefix containing the decision: with
  Corr(r_i, r_j) = κ the update becomes (G − 1)π(1 − π)(1 − κ)ΔQ, which changes only the coefficient.
- Off-policy data (importance weights): the identity needs on-policy samples. The G audit found an importance ESS of
  1.01 of 322 hinted episodes (`results/quality/g_mismatch_s0_ck100.json`), so hinted rollouts cannot be reweighted
  into this estimator.

---

## 3. Results specific to std normalisation

### Theorem 3.1 (shared state ⇒ per-state scale invariance)
Under the shared law and ε → 0, the expected GRPO update is invariant to every per-state positive affine transform of
the return, R ↦ λ_z R + κ_z (λ_z > 0). This holds for any policy class and any trajectory structure (single or
multiple decisions, stochastic or deterministic outcomes within the state). Dr. GRPO's update is not invariant: it
becomes (G − 1)·Σ_z w(z) λ_z ∇J_z.

*Proof.*
1. In a shared-state group every R_i is drawn from state z's law.
2. Group standardisation is invariant to a common positive affine map: (λR_i + κ − λR̄ − κ)/(λσ̂) =
   (R_i − R̄)/σ̂. So the conditional expected update given z is unchanged.
3. The unconditional update is Σ_z w(z)·(that).
4. For ε > 0 the invariance holds up to a factor λσ̂/(λσ̂ + ε). It fails only for groups whose spread is of order
   ε/λ_z. ∎

The stakes of state z (λ_z) therefore cannot reach the update. Only the *shape* of each state's return function and
the state's *probability* do. The Bayes-optimal policy maximises Σ_z w(z) λ_z J_z(θ) and does depend on λ.

### Corollary 3.2 (the stake-free vote; exact finite G)
Assume, in addition, **deterministic within-state returns**, R = α_z + β_z r. This holds when s ∈ {0, 1} or
repairability is part of the state, there are no exogenous failures, and there are no false alarms. Let
m_G(π) = E[√(k(G − k)/(G(G − 1))); 0 < k < G], with k ~ Bin(G, π). Then

  U_GRPO(π) = (G − 1)·m_G(π)·𝒱 + O(ε),  𝒱 = Σ_z w(z) sign(β_z).

*Proof.*
1. Given z: d_i = β_z(r_i − r̄) and σ̂_R = |β_z| σ̂_r.
2. Hence Σ A_i (r_i − π) = sign(β_z) Σ (r_i − r̄)(r_i − π)/σ̂_r = sign(β_z)(G − 1)σ̂_r, using
   Σ(r_i − r̄) = 0.
3. Take expectations over k, then over z. ∎

Since m_G(π) > 0 on (0, 1), **sign U_GRPO = sign 𝒱 at every π**. The flow goes to the corner selected by the vote,
whatever ΔQ is.

**The threshold p·s > 1/2: exact scope.** Canonical problem with repairability as part of the state:
- states clean (1 − p), faulted-repairable (p·s) and faulted-unrepairable (p(1 − s));
- β_clean = −c_clean, β_rep = V + L − c_rep, β_unrep = −c_unrep.

If 0 < c_rep < V + L and c_clean, c_unrep > 0, then 𝒱 = ps − (1 − ps) = 2ps − 1. So:
- **Sufficiency and necessity.** GRPO's expected update is positive at every π iff p·s > 1/2, negative iff
  p·s < 1/2.
- **Boundary.** At p·s = 1/2 the expected update is identically 0 (pure drift-free noise).
- **"Irrespective of read costs" holds exactly under four conditions:**
  - (a) each cost is positive;
  - (b) the cost is below the stake in the repairable-fault state;
  - (c) costs are large relative to ε/σ̂_r (c ≳ 10⁻³ at ε = 10⁻⁴; [scan] `cost_blindness`);
  - (d) within-state returns are deterministic.
- **How each extension fares:**
  - State-dependent costs, nonlinear costs of a single read, and penalties for unnecessary reads (added to c_clean)
    change nothing: only signs enter.
  - A cost above the stake flips that state's vote; this is the only route by which c enters.
- **When repair success is rollout randomness** (the LLM case and TB1's `s`), the returns within a faulted state are
  not deterministic. The vote is replaced by the general expression of Proposition 2.1. The GRPO threshold is no longer
  1/(2s): it lies between 1/2 and 1/(2s) and depends on c ([scan] `threshold_exactness`). **So the phrase "irrespective of costs" is false for s ∈ (0,1)
  as rollout noise.**

### Corollary 3.3 (endogenous homogeneity: the always-verify corner is governed by the vote under *every* law)
Suppose verifying makes the return state-independent, R(z, 1, o) ≡ R* (perfect detection, certain repair, no
exogenous failure), and skip returns are deterministic within a state, R(z, 0) = α_z. Then for **every** exchangeable
group law, ε → 0:

  b_{G−1} = (G − 1)/√G · 𝒱,  U′(1) = −(G − 1)√G · 𝒱.

*Proof.*
1. With G − 1 readers at R* and one skipper j at α = α_{z_j}: d_skip = (G − 1)(α − R*)/G, each reader's
   d = −(α − R*)/G, and σ̂ = |α − R*|/√G.
2. So Σ_{readers} A_i = −(G − 1)/√G · sign(α − R*) = (G − 1)/√G · sign(β_{z_j}).
3. The skipper's state has marginal w under every exchangeable law (A1, A3). ∎

**Meaning.** Near "always verify" the verifying rollouts' returns no longer depend on the hidden state. Whatever the
sampler did to decorrelate states, a group is then effectively homogeneous, and each lone skipper is standardised to
±(G − 1)/√G. **Resampling hidden states independently within the group does not remove the stake-free vote near
the corner.** When 𝒱 < 0 < ΔQ (p < 1/2 with cheap, valuable reads), always-verify is *unstable* for every group law;
only centred estimators reach it.

Under iid states the never-verify corner can still be unstable (b₁ > 0) once G is large enough. GRPO then converges to
an **interior** read rate π* ∈ (0, 1), with regret ΔQ·(1 − π*). [scan] `iid_interior` reports π* by G. Example: p =
0.3, c = 0.05, G = 8 gives π* ≈ 0.59, regret ≈ 0.10 (`test_iid_interior_equilibrium`).

**Endogenous homogeneity is an implementation fact in ActionProof:** a correct read plus repair gives R_V = 1 in
every world. `scripts/theory/llm_mechanism.py` §6 measures how far the trained agents' readers are from this
(MECHANISM_DIAGNOSTICS.md).

### Proposition 3.4 (the never-verify corner and the critical correlation)
- For mix(ρ), b₁(ρ) = ρ·b₁^{shared} + (1 − ρ)·b₁^{iid} (linearity in the law).
- b₁^{shared} = (G − 1)/√G · 𝒱 (same algebra as 3.3 with one reader).
- b₁^{iid} is a finite sum over the composition of the G − 1 skippers' states.
- If b₁^{iid} > 0 > b₁^{shared}, never-verify is locally stable iff ρ > ρ* = b₁^{iid}/(b₁^{iid} − b₁^{shared}).
- [scan] `critical_rho` reports ρ*(G) for mix and dirmult, and √G·ρ*.

### Proposition 3.5 (pairwise correlation is not a sufficient statistic)
- The mixture is linear: U_mix(ρ) = ρU_shared + (1 − ρ)U_iid, exactly.
- mix(ρ) and dirmult(ρ) have the same marginal and the same pairwise ICC ρ.
- For G = 2 and binary z they are the same law, so the updates coincide.
- For G ≥ 3 they differ in general (`test_pairwise_correlation_not_sufficient`). The update depends on the whole
  composition law Q(m), in particular on the probability that a group is payoff-homogeneous, not on ρ alone.
- [scan] `rho_curves` compares endpoints by family; TH1 tests the difference on held-out twins.

### Proposition 3.6 (large-G limit under the shared law)
- As G → ∞, σ̂_z → σ_z(θ), the std of R in state z under π_θ.
- Then U_GRPO/(G − 1) → Σ_z w(z) ∇_θ J_z(θ)/σ_z(θ). For a binary decision this is
  Σ_z w(z) π(1 − π) β_z/σ_z(π).
- GRPO therefore ascends a **per-state signal-to-noise** field: each state's gradient is divided by its own reward
  spread. Stakes cancel when the state's spread is driven by the decision (σ_z ∝ |β_z|).
- Cost sensitivity returns when another noise source dominates σ_z: exogenous failure, stochastic recovery, or other
  sampled decisions in a multi-step trajectory.
- This is the shared-state analogue of:
  - Davis and Recht's per-prompt arcsin√p transform (binary rewards, per prompt; arXiv 2510.13651);
  - Bereket and Leskovec's per-outcome 1/σ weighting (arXiv 2508.11800).
- The proof is the law of large numbers for σ̂ plus Slutsky's lemma; the finite-G bias is O(1/G).

---

## 4. Consequences for other estimators

| Estimator | Expected update under the shared law | Status |
|---|---|---|
| Dr. GRPO, RLOO | (G − 1)π(1 − π)ΔQ, resp. Gπ(1 − π)ΔQ, every law | Prop. 2.2 |
| **AWRB** (return := E[R \| own observations] under the true prior), then group std | In the canonical problem with s = 1 and perfect detection, *every* state has β = ΔQ = p(V + L) − c, so the vote is sign ΔQ (`test_awrb_vote_is_sign_of_delta_q`). With s < 1 as rollout noise, the clean state's β stays p(V + L) − c, larger than ΔQ = ps(V + L) − c. So AWRB-GRPO over-verifies when ps(V + L) < c < p(V + L). This is consistent with the v2 audit's "AWRB verifies in 18/18 mirror cells where VOI < 0" (verifying that those cells have s < 1 is listed in THEOREM_AUDIT §5) | Prop. 4.2 |
| **GDPO-style per-component normalisation** of (utility, −c·r) | The cost component is a full anti-read vote −(G − 1)σ̂_r in every non-degenerate group. The utility component is 0 in the clean state and +(G − 1)σ̂_r in the repairable fault. So U = −w(clean)(G − 1)m_G(π) ≤ 0: **never verifies, for any p < 1** (`test_gdpo_never_verifies_when_cost_is_a_component`). Consistent with TB1 GDPO wrong in 370/495 runs | Prop. 4.3 |
| **DAPO dynamic sampling** (drop zero-variance groups and refill) | Degenerate groups contribute 0, so the expected update per kept group is U/P(non-degenerate). Same sign: the filter changes step size, not direction | Prop. 4.4 |
| **External scale** (constant, batch std with B → ∞, EMA of past steps, per-task-family scale) | If the scale S is independent of the group's own states given the context, E[A·score] = E[1/S]·U_Dr, so the sign is ΔQ. Batch std over B groups depends on the group's states through 1/B of its terms (bias O(1/B)) | v2 normaliser condition; LAB1 H4 (causal, supported) |
| **PPO** (learned value baseline, batch advantage normalisation, clipped epochs) | First epoch on-policy: ratio 1, sign ΔQ; later epochs clip magnitudes, never signs of the per-sample terms | §5 counterexamples; TH1 descriptive |
| **KL penalty** β·KL(π‖π_ref) | Adds −β·π(1 − π)(θ − θ_ref) per rollout; moves the corners inward, does not change the vote's sign away from π_ref | TH1 `grpo_kl` |
| Token-length normalisation (`loss_type="grpo"`, 1/\|τ_i\|) | Under the shared law each state's rollouts are reweighted by positive factors: coefficients change, signs do not. **ActionProof uses `dapo`,** so this is not the trained setting | COUNTEREXAMPLES.md |

---

## 5. Assumption register

| # | Assumption | Needed by | Holds in ActionProof AD training? | If violated |
|---|---|---|---|---|
| A1 | Decision aliased (independent of z given h) | all | yes at the first read after an ok ack (identical observations by construction, `CWEnv` ids) | results hold per observable history |
| A2 | Rollouts conditionally independent given states | all | yes (independent vLLM samples, T = 1) | coefficient (1 − κ) |
| A3 | Exchangeable group law with marginal w | all | yes: shared law | — |
| S | Shared law | 3.1, 3.2 | **yes** | mixtures: Prop. 3.4/3.5; iid: Cor. 3.3 still applies near the corner |
| D | Deterministic within-state returns | 3.2 (vote form), 3.3 | **no**: later sampled actions (repair, report) make returns random within a state | vote replaced by Prop. 2.1 / 3.6; threshold moves off 1/(2s) and depends on c |
| N | Reading neutralises the state | 3.3 | approximately (read + repair ⇒ R_V = 1); measured in MECHANISM_DIAGNOSTICS | b_{G−1} no longer law-invariant |
| E | c ≫ ε/σ̂_r | 3.2 cost-blindness | yes (c = 0.05 vs ε = 1e-4) | threshold falls below 1/2 at c ≲ 10⁻³ |
| B | Single binary decision | 3.2, 3.3, 3.4 | no (several reads per trajectory) | 3.1 and 3.6 still hold; vote is an approximation |
