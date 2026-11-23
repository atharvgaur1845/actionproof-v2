# Candidate C: posterior-CVaR returns (risk-sensitive, substantially different objective)

Status: implemented as estimator `cvar` (`actionproof/ad/worlds.py:marginalise` returns `cvar`). **Not scheduled**
in AD-D1: the 2-GPU budget goes to A and B first. It would be registered separately.

- **Objective.** J_α = E_τ[CVaR_α(R(τ, w) | τ)] − c·E[reads].
  - CVaR_α is the mean of R_V over the worst α share of the posterior mass of the worlds consistent with τ.
  - α = 0.25 in the code.
  - This is a different objective from E[R]: it prices the hidden-world risk that a trajectory leaves unresolved.
  - A trajectory that writes and never reads keeps every world consistent, including the faulted ones, so its CVaR
    return is near 0.
  - A trajectory that reads and confirms gets its revealed return.
- **What it should do.** It verifies wherever the hidden world could still differ (pay, order, receive). It does not
  verify where it cannot (the journal entry under cw1). It stays risk-averse even at low p, which suits irreversible
  operations.
- **Failure modes.**
  - Over-verification relative to the expected-value optimum. That is by design, so it must be judged on a
    risk-adjusted metric, not on J_c.
  - With group std it inherits the shared-world reweighting.
  - With α → 0 it becomes "always verify every faultable write".
- **Novelty.** Weak. CVaR policy gradients (Tamar et al.) and risk-averse Bayes-adaptive RL (Rigter et al., NeurIPS
  2021, arXiv 2102.05762) already do this; the per-trajectory replay posterior is the only new piece.
- **Falsification experiment (if run).** At c = 0.30 the expected-value optimum is "never verify". C should still
  verify after payments, and lose J_c while cutting unverified-fault episodes. If it does not verify, the posterior
  CVaR signal is not reaching the decision.
