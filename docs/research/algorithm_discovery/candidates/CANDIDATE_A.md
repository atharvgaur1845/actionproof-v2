# Candidate A: AWRB, aliased-world Rao-Blackwellised group advantages

Status: implemented (`actionproof/ad/worlds.py`, `rewards.py`); development round AD-D1 registered
(EXPERIMENT_PROTOCOL.md and Amendment 1). Two scalings are tested: `awrb` (group std) and `awrbn` (no std).

## 1. Setting and notation

- Task x, with requested steps and initial state s₀(x).
- **Hidden world** w ∈ W(x): the fault plan, i.e. which write attempt fails silently and how. It is drawn from a
  known prior π₀(w | x) that is independent of the prompt text.
- **Policy** π_θ(a_t | h_t), where h_t = (x, a_{<t}, o_{<t}) is the visible history.
- **Observations** o_t = O(s_t, a_t; w). The simulator is deterministic given (x, w, the action sequence, the id seed).
- **Trajectory** τ = (a_{1:T}, o_{1:T}); return R(τ, w) = R_V(final state) − c · #reads(τ).

Since π_θ depends on observations only,

  p_θ(τ | x, w) = Π_t π_θ(a_t | h_t) · 1[o_t = O(·; w)].

So for any two worlds w, w′ that produce the same observations along τ, p_θ(τ | x, w) = p_θ(τ | x, w′). Call this
equivalence class 𝒲(τ).

## 2. Estimator

**Rao-Blackwellised return.**

  R̄(τ) = E[R | τ, x] = Σ_{w ∈ 𝒲(τ)} π₀(w | x) R(τ, w) / Σ_{w ∈ 𝒲(τ)} π₀(w | x).

It is computed exactly by replaying τ's own calls in every w ∈ supp π₀. A world is kept iff every observation
matches. Its R_V is scored on the replayed final state.

**Facts (all exact, tested in `tests/ad/test_ad.py` by full enumeration):**
1. E_{w,τ}[R̄] = E_{w,τ}[R]. This is the tower property; `test_unbiased_and_rao_blackwell`.
2. Var(R̄) ≤ Var(R) (Rao-Blackwell); same test.
3. ∇_θ log p_θ(τ) is τ-measurable, so E[∇ log p_θ(τ) R̄] = E[∇ log p_θ(τ) R] = ∇J. Plain REINFORCE with R̄ is
   unbiased (`test_exact_policy_gradient_identity`).
4. For a trajectory that never observes the world (no reads, no timeouts), 𝒲(τ) is the whole support and R̄ is the
   prior mean. For one that reads, 𝒲(τ) shrinks to the worlds consistent with what it read.

**Advantage.**
- `awrb`: A_i = (R̄_i − mean_g R̄) / (std_g R̄ + ε) (TRL `scale_rewards="group"`).
- `awrbn`: A_i = R̄_i − mean_g R̄ (`scale_rewards="none"`).

Everything else is TRL's GRPO loss.

## 3. Why the scaling matters (AD-T0b; FINAL_ALGORITHM.md §2)

Group rollouts share w. Write σ_g for the group std.
- **Raw returns with group std (standard GRPO).** The expected update on "read" is Σ_w π₀(w) (1/σ_w) E[(R − μ_w) | read, w].
  Each world is weighted by 1/σ_w, not by its reward scale. That is L1's mechanism.
  - In clean worlds the only within-group difference is the read cost. Std scaling turns that tiny difference into a
    unit advantage against reading.
  - With one decision and perfect repair, the per-world terms are ±sqrt((1−ρ)/ρ), so the sign is that of 2p − 1,
    whatever c is.
- **R̄ with group std (`awrb`).** No-read rollouts get the prior mean in every world. Read rollouts keep their
  revealed-world returns.
  - In clean worlds, reading now beats not reading by p − c.
  - In faulted worlds with failed repair, reading loses.
  - Std scaling again equalises the worlds, so the sign is pushed toward reading for p < 1/2 even when reading is
    worthless (AD-T0b: reads at s = 0).
- **R̄ or R with no std (`awrbn`, `nostd`).** The update is proportional to ∇J in expectation: the correct VoI sign.
  - R̄ removes world noise from no-read rollouts: higher gradient SNR when repair succeeds often (AD-T0b: 10.7 vs
    1.08 at s = 1).
  - It breaks the shared-world pairing that raw within-group comparisons enjoy: lower SNR when repair never succeeds
    (s = 0).

## 4. Pseudocode (per training step)

```
for each prompt x in batch:                       # 4 prompts
    w* = x.meta.faults                            # the group's world (unknown to the policy)
    for i in 1..G:                                # G = 8
        tau_i = rollout(pi_theta, env(x, w*, seed_i))
        cons = [(pi0(w), R_V(replay(tau_i.calls, env(x, w, seed_i)))) for w in supp(pi0(.|x))
                if replay reproduces every observation of tau_i]
        Rbar_i = sum(p * r for p, r in cons) / sum(p for p, _ in cons) - c * reads(tau_i)
    A_i = Rbar_i - mean(Rbar)                     # awrbn; awrb also divides by std(Rbar) + 1e-4
GRPO/DAPO token loss with A_i on policy tokens, KL beta 0.02 to the warm start
```

**Complexity.**
- Per trajectory: |supp π₀| replays × #calls. That is 8–14 worlds × ≤ 16 calls of pure-Python dict operations,
  about 1 s per step on the CPU, against tens of seconds of generation.
- No extra generation.
- No extra forward passes.

## 5. Implementation details that matter

- **Call-indexed ids (`CWEnv`).** A dup_delivery's hidden second row must not shift the ids that later calls show.
  Otherwise worlds that differ only in an unobserved fault would be spuriously distinguishable. The test
  `test_write_acks_identical_across_silent_worlds` checks this.
- **The prior must equal the pool's generating distribution.** `pools.py` draws from `world_prior`, and
  `test_pool_worlds_are_in_prior` checks it. A misspecified prior biases R̄ toward the assumed prior's expectation.
  This is an open ablation.
- **Visible timeouts.** lost_ack and precommit_timeout are aliased with each other, but distinguishable from ok
  acknowledgements. Replay handles this with no special cases.

## 6. Anticipated novelty objections

1. **"This is Rao-Blackwellisation / input-dependent baselines / SAMPLR."** The statistical device is classic. The
   claim is narrower:
   - exact conditioning of an LLM agent's return on the observation-equivalence class of hidden execution worlds,
     computed by replay with zero extra rollouts;
   - the analysis of how it interacts with group standardisation, including the mirror bias, which we did not
     expect.
2. **"Dr. GRPO already removes the std."** Yes. Candidate B is exactly that, and it is the baseline A must beat
   through variance reduction. If awrbn ≈ nostd, A has no empirical contribution beyond the diagnosis.
3. **"It needs a known fault prior."** True. It is a training-time simulator assumption, like SAMPLR's. Deployment
   needs nothing.

## 7. Falsification criteria

- **Registered (EXPERIMENT_PROTOCOL.md §4 and Amendment 1).** awrbn fails if any of these holds:
  - VSC on cw_dev is not ≥ grpo + 0.03 in the mean of 2 seeds, with each seed positive;
  - J_c is below grpo's;
  - duplicates are more than +0.02;
  - the KL ratio is more than 10×.
- **The claim "R̄ adds beyond the std fix" is falsified at development level if awrbn does not beat nostd on VSC or
  J_c, or on training-signal stability.**
- **The mirror-bias prediction is falsified if, at c = 0.30, awrb's verify-after-write rate is not above awrbn's.**
