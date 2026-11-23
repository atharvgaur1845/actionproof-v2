# Candidate B: no-std group advantages for costly verification (the known fix, lower risk)

Status: implemented as estimator `nostd` (TRL `scale_rewards="none"`). Registered in AD-D1.

- **Formulation.** The same task, world prior, reward R_c = R_V − c·reads and trainer as everything else in AD-D1.
  The advantage is A_i = R_i − mean_g R: no division by the group std. The token loss aggregation is TRL's default,
  as in every ActionProof arm; Dr. GRPO's length-normaliser change is not applied, so exactly one thing differs.
- **Why it should work.** Without std scaling, the expected update is ((G−1)/G)·∇J up to clipping and KL (Bereket and
  Leskovec, arXiv 2508.11800). So the verification decision follows the VoI sign p·s − c.
- **Novelty.** None as a method (Dr. GRPO, arXiv 2503.20783). The contribution, if it works, is the diagnosis:
  - group-standardised GRPO trains a state-rewarded agent to trust acknowledgements whenever faults are rarer than
    a coin flip;
  - the remedy is already in the literature, and verification can be restored by removing one normalisation.
- **Expected failure modes.**
  - Slower learning: advantages are a factor ~1/σ smaller.
  - World-luck variance.
  - Reweighting of every task by its reward variance (Dr. GRPO's difficulty argument), which could change partial
    repair learning elsewhere.
- **Ablation.** grpo vs nostd at identical everything is the ablation.
- **Falsification.** Gates G1–G4 vs grpo (EXPERIMENT_PROTOCOL.md §4). Prediction P1: verify-after-pay rises above
  the warm start's under nostd and falls under grpo.
