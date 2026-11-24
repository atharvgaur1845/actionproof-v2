# Partially shared hidden states: what happens between "one state per group" and "independent states"

Evidence level: **[math]** exact enumeration (`actionproof/theory/exact.py`), numbers from `results/theory/scan.json`.
Canonical problem unless stated: one fault state with prior p, s = 1, d = 1, V = 1, L = 0, c = 0.05, ε = 1e-4.

## 1. Constructions and their correlation

- **mix(ρ):** with probability ρ the group shares one draw z ~ w; otherwise each rollout draws independently.
- **dirmult(ρ):** z_i | θ ~ θ i.i.d., with θ ~ Dirichlet(((1 − ρ)/ρ)·w).

Both have marginal w and pairwise intra-class correlation **exactly ρ** for every state indicator
(`pairwise_icc`, tested to 1e-9). The mixture's induced pairwise correlation is
Cov = ρ·w(1 − w) + (1 − ρ)·0, hence Corr = ρ.

## 2. Closed forms that exist

| Quantity | Closed form? | Expression |
|---|---|---|
| Dr. GRPO expected update | **yes, for every law** | (G − 1)π(1 − π)ΔQ (Prop. 2.2): **independent of ρ** |
| GRPO expected update | yes as a finite sum | U(π) = Σ_k C(G,k)π^k(1 − π)^{G−k} b_k(law) (Prop. 2.1). For the mixture **exactly linear in ρ**: U = ρU_shared + (1 − ρ)U_iid. For dirmult, a finite sum over compositions; not linear in ρ |
| Shared-law part | yes | U_shared = (G − 1)m_G(π)·𝒱 |
| Corner stability | yes | U′(0) = G b₁, U′(1) = −G b_{G−1} |
| Always-verify corner, any law, read neutralises state | **yes, law-free** | U′(1) = −(G − 1)√G·𝒱 (Cor. 3.3) |
| Never-verify corner | mixture: ρ* = b₁^{iid}/(b₁^{iid} − b₁^{shared}); dirmult: root of a finite sum | Prop. 3.4 |
| Decision boundary in p | for each law, the zero of a degree-G polynomial in π, at fixed π; no elementary closed form for iid/dirmult | — |
| Regret of the learned policy | ΔQ·\|π_∞ − π*_Bayes\| with π_∞ the first zero of U in the direction of sign U(π₀) | single decision: J is linear in π |

The interpolation is **not** linear in the quantities that matter. U is linear in ρ for the mixture, but the endpoint
and the regret are not: the endpoint jumps from an interior point to 0 at ρ* (below).

## 3. Results

### 3.1 Independent states do not cure GRPO (endogenous homogeneity)
Read neutralises the state (s = d = 1). So near always-verify every group is payoff-homogeneous, and U′(1) has the
sign of −𝒱 = 1 − 2p for every law. For p < 1/2 always-verify is unstable for every law and every G.

| p | G | iid endpoint π* (from 0.5) | G(1 − π*) | regret | never-verify stable under iid? |
|---|---|---|---|---|---|
| 0.1 | 8 / 32 / 64 | 0 / 0 / 0.684 | — / — / 20.3 | 0.050 / 0.050 / 0.016 | yes up to G = 32 |
| 0.2 | 8 / 16 / 64 | 0 / 0.594 / 0.914 | — / 6.50 / 5.52 | 0.150 / 0.061 / 0.013 | yes up to G = 8 |
| 0.3 | 4 / 8 / 16 / 64 | 0 / 0.597 / 0.831 / 0.962 | — / 3.23 / 2.70 / 2.43 | 0.250 / 0.101 / 0.042 / 0.0095 | yes up to G = 4 |
| 0.4 | 8 / 16 / 64 | 0.844 / 0.933 / 0.985 | 1.25 / 1.07 / 0.96 | 0.055 / 0.023 / 0.0052 | yes at G = 2 |
| 0.45 | 8 / 64 | 0.928 / 0.993 | 0.58 / 0.43 | 0.029 / 0.0027 | yes at G = 2 |

(`iid_interior`, c = 0.05.)

**Corollary (scaling, numerical).** G(1 − π*) converges as G grows: to ≈ 2.4 at p = 0.3, ≈ 0.95 at p = 0.4, ≈ 0.43 at
p = 0.45. So 1 − π* = Θ(1/G) and the regret is Θ(1/G) under independent states, against Θ(1) under the shared law
(regret = ΔQ when 𝒱 < 0 < ΔQ). Intuitively, the vote dominates once the expected number of skippers in a group is
O(1). This is established numerically to G = 64, not proved.

**The 1/G scaling does not apply to rare faults.** For p ≤ 0.1, even iid groups keep never-verify stable up to G = 32:
all-clean groups have probability (1 − p)^{G−1} and their lone reader gets a standardised advantage of −(G − 1)/√G.

### 3.2 A small shared fraction restores the full collapse
Mixture: never-verify is locally stable iff ρ > ρ* = b₁^{iid}/(b₁^{iid} − b₁^{shared}) (`critical_rho`).

| p | G = 8 | G = 16 | G = 32 | G = 64 | √G·ρ* at G = 64 |
|---|---|---|---|---|---|
| 0.2 | 0 (iid already stable) | 0.084 | 0.097 | 0.073 | 0.58 |
| 0.3 | 0.211 | 0.247 | 0.196 | 0.149 | 1.19 |
| 0.4 | 0.532 | 0.476 | 0.395 | 0.319 | 2.55 |

- For the mixture, b₁^{shared} = −(G − 1)/√G·|𝒱| grows like √G while b₁^{iid} tends to a constant. So
  ρ* ~ b₁^{iid}(∞)/(|𝒱|√G) → 0 as G^{-1/2}. The √G·ρ* column is still rising slowly towards that limit
  (1.34 for p = 0.3).
- **For the Dirichlet-multinomial, ρ* does not vanish:** 0.152 / 0.244 / 0.284 / 0.305 at G = 8 / 16 / 32 / 64
  (p = 0.3).

### 3.3 Pairwise correlation is not sufficient
- mix(ρ) and dirmult(ρ) have identical marginals and pairwise correlation, but their critical correlations differ:
  ρ* = 0.149 vs 0.305 at G = 64, p = 0.3.
- Their updates are identical at G = 2 and differ for G ≥ 3 (`test_pairwise_correlation_not_sufficient`).
- What matters is the probability that a group is **payoff-homogeneous**:
  - the mixture puts mass ρ on exactly homogeneous groups at every G;
  - the Dirichlet-multinomial's homogeneous mass decays with G at fixed ρ.
- Practical meaning: a sampler cannot be summarised by one "state correlation" number. Two samplers with the same
  correlation can produce different learned verification rates.

### 3.4 Group size
- Under the shared law the vote is G-independent in sign. G changes only the step size, through m_G.
- Under partial sharing, larger G **helps iid groups** (Θ(1/G) regret) but **makes the shared fraction more
  dominant** (√G amplification of a lone deviator in a homogeneous group).
- The two effects cross. The stability of never-verify is non-monotone in G at fixed ρ: for p = 0.3, ρ* = 0.211 / 0.254 / 0.196 at G = 8 / 12 / 32, so ρ = 0.22 collapses at G = 8 and 32 but not at G = 12.

## 4. What this means for real rollout samplers

| Sampler | Group law | Prediction |
|---|---|---|
| Same prompt, environment state fixed per task (ActionProof, and most task-instance agent benchmarks where the database/repo/site snapshot is part of the task) | shared | full vote; severity invisible |
| Same prompt, state redrawn per rollout (V2-E1 design; never ran) | iid | interior read rate, regret Θ(1/G) for p not small; collapse for rare faults at G ≤ 32 |
| Paired / common-random-number rollouts (e.g. Sakib 2026's event-keyed noise schedules) | shared w.r.t. the paired noise | if combined with group std, the paired component is voted, not weighed (prediction; not tested on their setup) |
| Group by task family across instances | iid-like states plus instance-level offsets α | extra between-instance variance acts like exogenous noise and partly restores cost sensitivity (scan `exogenous_noise`), but the baseline is no longer per-prompt (higher variance) |

**Recommendation that follows from the mathematics.** Do not try to fix the bias by decorrelating states inside the
group: Cor. 3.3 shows it cannot fully work. Use a scale that does not depend on the group's own hidden states: no std
(Dr. GRPO, RLOO), a batch- or task-family-level scale, or an EMA. All of these preserve sign ΔQ (Prop. 2.2 and the
normaliser condition) **in expectation**. The EMA-scaled LLM arms (AD-D2) still collapsed to 0 reads in one of two
seeds each (`results/ad/d2_summary.json`), so the expected-update argument does not guarantee stable training. Independent resampling remains useful for variance, not for bias.
