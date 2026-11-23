# Algorithm candidates (v2)

Each candidate below is stated with:
- one hypothesis;
- the update it uses;
- the closest prior method;
- the information it needs;
- its tests;
- its smoke check;
- the registered evidence, and a decision.

Code is in `actionproof/ad2/tabular/train.py`; registration in `V2TB_PROTOCOL.md`. Notation follows
GRPO_FAILURE_DERIVATION.md. b(z | h) is the posterior over hidden fault plans given the observable history.

## Direction A: belief-marginalised returns (AWRB, AWRB-n)

- **Hypothesis A1.** Replacing R(τ, z) by R̄(τ) = Σ_{z ∈ Z(τ)} b(z | τ)·R(τ, z) lowers the variance of a correctly
  scaled policy gradient enough to learn verification faster than nostd at the same budget.
- **Update.**

      Â_i = R̄_i − mean_g R̄

  This is unbiased (AWRB audit Q6).
- **Closest prior.** Input-driven baselines (Mao et al. 2019); SAMPLR; paired rollouts (Sakib 2026).
- **Information.** Simulator plus prior weights.
- **Tests.**
  - `test_rao_blackwell_unbiased_and_lower_variance`;
  - `test_consistent_worlds_reproduce_the_observations`.
- **Evidence.**

  | Setting | awrbn | nostd | Reading |
  |---|---|---|---|
  | TB1 regret | 0.0123 | 0.0132 | about equal |
  | TB3 test_shift | 0.0136 | 0.0151 | about equal |
  | TB4, true prior | 0.0295 | 0.0347 | about equal |
  | TB4, ×0.25 prior | 0.260 | — | catastrophic |
  | LLM cw_dev | worse in both seeds | — | — |

- **Decision. Rejected as a contribution.** The variance reduction exists but is small. Once the policy reads,
  R̄ = R (the posterior collapses), and the gain does not survive prior misspecification.
- **Group-std variant (awrb).** Rejected for the mirror bias (TB1: verifies in 18/18 worthless-read cells).

## Direction B: cost-calibrated advantage scaling

- **Hypothesis B1.** A scale that is independent of the current samples' hidden states keeps GRPO's
  adaptive-step benefit (arXiv 2601.23135) without its bias. Two forms:
  - `ema`: a global EMA of past group stds (first-stage D2);
  - `ctxema`: one EMA per observable context.
- **Update.**

      Â_i = (R_i − mean_g R) / s_{ctx, t−1}

  where s is computed from earlier steps only.
- **Closest prior.** PPO's running reward normalisation; per-prompt statistics.
- **Information.** None.
- **Evidence.**
  - **TB1:** ema and ctxema are sign-correct (wrong-decision rates 0.048 and 0.071, vs nostd 0.057).
  - **TB3:** ctxema ×1 is *worse* than nostd ×1 (test_shift 0.0267 vs 0.0151) and better at ×4 (0.0048 vs
    0.0045, about equal).
  - **LLM D2:** nostd_ema and awrbn_ema each collapse to 0 reads in one of two seeds; both fail G1.
- **Decision. Rejected.** Sign preservation holds, as derived, but no benefit over nostd was shown. In the LLM,
  GRPO-sized steps on a noisy, sign-correct signal fell into the absorbing ρ = 0 state in 2 of 4 runs. The
  derivation's normaliser-independence condition is the contribution; these estimators are its illustrations.

## Direction C: learned verification-budget allocation (Lagrangian)

- **Hypothesis C1.** Dual ascent on λ for E[c·reads] ≤ B lets the policy allocate a budget across risky operations
  better than a fixed penalty.
- **Update.**

      R_λ = U − fail − dup − λ·c·n_reads
      λ ← max(0, λ + η(E[c·n] − B))

  With group std this is `lagr_grpo`; with centring only it is `lagr`.
- **Closest prior.** CPO; RCPO; Constrained GRPO.
- **Information.** None.
- **Evidence.**
  - TB1 `lagr_grpo`: wrong-decision rate 0.628, regret 0.124. λ cancels inside cost-only groups (derivation §3d),
    and a budget cannot rescue group-std GRPO.
  - `lagr` (centring) is in TB2's baseline pool.
- **Decision. Rejected as a fix under group std**, as derived.
- **Remaining question.** Whether a *state-conditioned* budget beats a penalty when the penalty is set correctly.
  At the optimum the Lagrangian with B = the oracle's cost is the penalty problem, so a gain could come only from
  optimisation dynamics, which was not demonstrated.

## Direction D: counterfactual decision-level credit

### D-i. `igc`: information-gated credit

- **Hypothesis.** Giving every *provably uninformative* verification decision its exact all-action value removes
  the sequence-level smear that prevents selectivity (first-stage F3). A decision is uninformative when its read
  observation is the same in every hidden state of the prior's support that is consistent with h; its exact value
  is then Q(verify) − Q(skip) = −c.
- **Update.**

      g_t = π(1 − π)·(−c)·φ_t

  for such decisions. Every other decision gets the nostd sequence term.
- **Closest prior.** First-stage D3 (registered, run on the LLM as `nostd_ema_irc`); per-step credit methods.
  Unlike C5 (main session, failed), it is not normalised across turns.
- **Information.** The prior's **support** only, plus replay.
- **Evidence.**
  - **TB1:** wrong-decision rate 0.034, regret 0.0060; no extra episodes.
  - **TB4:** regret 0.0121, against nostd's 0.0347, and robust to wrong weights. It loses its advantage when the
    support is wrong.
  - **TB3:** test_shift 0.0114 (×1) vs nostd 0.0151 (×1), and 0.0041 (×4) vs 0.0045 (×4).
  - **TB2:** see §Results.

### D-ii. `fork_act`: actual-world forks

- **Hypothesis.** At a verification decision h, K paired continuations per action, forked in the *actual* hidden
  world with common random numbers, estimate the belief-level Q-difference without bias. The actual world given h
  is a posterior draw. This gives the right verification decision at h with far less noise than a sequence-level
  return.
- **Update.** The all-action gradient

      Σ_a π(a | h)·(Q̂(h, a) − Q̄)·∇log π(a | h)

  with no normaliser.
- **Closest prior.** VinePPO (forks for value estimates); Tree-GRPO and TreeRL; mean actor-critic.
- **Information.** A simulator snapshot.

### D-iii. `fork_bel` and `cvc`: belief forks, and belief forks with exact local credit

- **Hypothesis.** Drawing each fork's hidden world from b(z | h) lowers the across-world variance of D-ii. It also
  fixes AWRB's trajectory-consistency limitation, because each alternative world gets its own *re-generated*
  continuation.
- **cvc.** Uses igc's exact value where a read is uninformative (no forks), and belief forks only where a read is
  informative.
- **Information.** Snapshot plus prior weights.
- **Tests.**
  - `test_belief_fork_estimates_belief_voi`: E[Q̂(v) − Q̂(s)] ≈ p·s − c in a clean actual world; the actual-world
    fork gives −c there, as it should;
  - `fork_bel_std` is the diagnostic for the normaliser condition.

### Evidence for D-ii and D-iii (before TB2)

| Testbed | fork_act | fork_bel | cvc | Reading |
|---|---|---|---|---|
| TB1 regret (not budget-matched) | 0.0072 | 0.0060 | **0.0043** | best of all 18 arms; wrong-decision rate 0.012 |
| TB3 (budget-matched) | — | — | — | every fork method has 2–3× the regret of nostd, igc or critic_q trained 4× longer on about the same number of episodes |
| TB4, ×0.25 prior | — | 0.256 | 0.254 | catastrophic |
| TB4, empirical-Bayes prior | — | — | 0.0096 | the failure is fixed |

## Results of the registered candidate gates (TB2)

Filled in from `results/ad2/tabular/TB2_summary.json`. The SFB is the strongest fair non-privileged baseline at a
matched episode budget.

| Variant | SFB (matched) | Regret, candidate vs SFB | Seed differences (SFB − candidate) | T1 | T2 | T3 | Pass |
|---|---|---|---|---|---|---|---|
| **igc ×1** | rloo ×1 | 0.0155 vs 0.0296 | +.015 +.011 +.018 +.015 +.012 | ✓ | ✓ | ✓ | **yes** |
| **igc ×4** | nostd ×4 | 0.0053 vs 0.0080 | +.0025 +.0028 +.0036 +.0028 +.0016 | ✓ | ✓ | ✓ | **yes** |
| igc ×8 | rloo ×8 | 0.0043 vs 0.0051 | +.0006 +.0011 +.0012 +.0010 −.0002 | ✗ (16% < 20%; 4/5 seeds) | ✓ | ✓ | no |
| cvc k1 | rloo ×8 | 0.0103 vs 0.0051 | all negative | ✗ | ✓ | ✓ | no |
| cvc k4 | rloo ×8 (under-budget) | 0.0093 vs 0.0051 | all negative | ✗ | ✓ | ✓ | no |
| fork_bel k1 / k4 | rloo ×8 | 0.0142 / 0.0129 vs 0.0051 | all negative | ✗ | ✓ | ✓ | no |
| fork_act k1 / k4 | rloo ×8 | 0.0167 / 0.0174 vs 0.0051 | all negative | ✗ | ✓ | ✓ | no |
| ctxema ×1 / ×4 / ×8 | rloo / nostd / rloo | worse at every budget | mostly negative | ✗ | ✓ | ✓ | no |
| awrbn ×1 / ×4 / ×8 | rloo / nostd / rloo | 0.0249 vs 0.0296; 0.0075 vs 0.0080; 0.0049 vs 0.0051 | mixed | ✗ | ✓ | ✓ | no |

## Ranking (novelty, feasibility, justification, improvement, falsifiability)

| Rank | Candidate | Novelty | Feasibility in LLM | Theory | Measured improvement | Falsified? | Decision |
|---|---|---|---|---|---|---|---|
| 1 | **igc** (information-gated exact local credit) | weak–moderate | high: replay already exists; LLM version = first-stage D3 | exact: an uninformative read's Q-difference is exactly −c | passes T1–T3 at 1× and 4×; selectivity 10× better at 1× | not falsified in the testbed; the LLM test (D3) is pending | **survives as the only candidate**; LLM evidence required |
| 2 | cvc | weak–moderate | medium: forks need env snapshots and extra generation | unbiased belief Q | best regret per *update*; fails at matched budget | **falsified** as cost-effective (TB2, TB3) | rejected |
| 3 | fork_bel / fork_act | weak | medium | unbiased | fails at matched budget | **falsified** | rejected |
| 4 | awrbn | moderate (first stage) | done | unbiased RB | ≈ nostd; catastrophic under p/4 | **falsified** as superior | rejected |
| 5 | ctxema | weak | high | sign-preserving | worse than nostd | **falsified** | rejected |
| 6 | lagr / lagr_grpo | none | high | cancellation under std | worse | **falsified** | rejected |

**Does any surviving candidate change a meaningful part of the optimisation procedure?** Yes: igc changes the
credit assigned to verification decisions (the advantage estimator), not the reward. It is still only a development
result in a tabular testbed. It needs privileged training-time information: the support of the hidden-fault prior
and a replayable simulator.
