# Candidate ranking (before any AD GPU result)

Scores run from 1 to 5, where 5 is best. For cost, 5 means cheap.

| Candidate | Novelty | Plausibility | Relevance | Expected gain | Clean evaluation | Generalisation | Cost | Decision |
|---|---|---|---|---|---|---|---|---|
| A: AWRB (awrbn; awrb kept as registered) | 3 | 4 | 5 | 3 | 5 (only the estimator differs) | 3 | 5 | **Run (AD-D1)** |
| B: no-std GRPO for costly verification | 1 | 5 | 5 | 4 | 5 | 3 | 5 | **Run (AD-D1)**: the strongest baseline for A |
| C: posterior-CVaR returns | 2 | 3 | 4 | 2 | 3 (different objective) | 3 | 5 | Implemented, deferred |
| D: certified invariance KL (renamed tools, workflow) | 2–3 | 3 | 5 | 2 (SR1 warns) | 4 | 4 | 2 | Deferred: needs pair construction for both shifts plus ≥ 6 runs; no budget this week |
| E: C5 turn credit / G realizable hints | 1 | 3 | 4 | ? | 4 | 2 | 4 | Running in the main session; results read, not edited |
| F: explicit VoI bonus | 1 | 2 | 4 | 2 | 3 | 2 | 5 | Rejected (double counting; reward shaping) |
| G: Lagrangian read budget | 1 | 2 | 4 | 2 | 3 | 2 | 4 | Rejected (published; the multiplier cancels under group std) |
| H: world-decoupled groups | 1 | 2 | 3 | 1 | 4 | 2 | 4 | Rejected by AD-T0 |

**Why A and B are run together.**
- B tests the diagnosis: group std makes verification cost-blind.
- A tests whether exploiting the simulator's world prior adds anything beyond the known fix.
- If B alone closes the gap, the honest contribution is the diagnosis, and A is reported as an unneeded refinement.

**Why the c = 0.30 intervention is in the first round.**
- It is the cheapest experiment that can falsify the most surprising prediction: that group-std AWRB over-verifies.
- It also checks that no-std estimators stop verifying when verification is not worth its price. This is the "not
  excessively conservative" requirement.
