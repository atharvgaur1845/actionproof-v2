# Research gap

## The gap

**Costly verification under group-relative policy optimisation.** In RL for tool agents, the agent's decision to
verify a write is a value-of-information decision. Verify iff the expected gain from detecting a silent failure,
P(fault | history) × P(repair), exceeds the verification cost c.

Two features of how such agents are trained in practice (TRL/verl GRPO-style trainers, multi-turn tool environments)
interact badly:

1. **Group-shared hidden worlds.** A group's rollouts are sampled from one task instance, and the hidden execution
   outcome (which write silently failed) is a property of that instance. In ActionProof it is literally fixed per task.
2. **Per-group reward standardisation.** GRPO's default divides by the group std.

Bereket and Leskovec (arXiv 2508.11800, 2025) showed that per-group std normalisation with a group-shared stochastic
outcome biases GRPO toward overconfident probability predictions. In a single-turn prediction task, no-std GRPO,
RLOO and PPO are calibrated.

**Not addressed in the literature I could find** (searches logged in `LITERATURE_AND_NOVELTY_AUDIT.md`):
- what this mechanism does to a sequential agent's *information-acquisition* decisions;
- whether an estimator can use the simulator's knowledge of the hidden-world prior to remove world luck without
  extra rollouts.

The cost-of-observation RL literature (ACNO-MDPs, Act-Then-Measure, AMRL) works with value-based or planning
methods, not with group-relative LLM policy optimisation. ActionProof's own registered work used free reads only.

**Technical limitation, stated as a testable claim.** With a group-shared hidden world and per-group std scaling, the
expected GRPO update on a verification action no longer has the sign of its value of information:
- with one aliased decision and perfect repair, it has the sign of 2p − 1, independent of the cost c;
- with imperfect repair, the threshold shifts further.

So a GRPO-trained agent with a state-grounded reward still learns to trust acknowledgements whenever faults are less
common than a coin flip, however cheap verification is. That is the deployment regime. ActionProof's earlier
"state reward teaches read-back" result holds only at c = 0, where clean groups have zero variance. This is a
property of the *optimiser*, not the reward. It is invisible in every earlier experiment.

## What would close it

- An advantage estimator whose expected update on verification has the VoI sign at any cost and fault rate, with
  lower variance than the obvious unbiased fix (dropping the std, as in Dr. GRPO).
- An LLM-agent demonstration that the pathology occurs, and that the fix restores selective verification:
  - verify where it pays;
  - skip verification where the hidden world cannot differ, e.g. an unfaulted service;
  - stop verifying when c exceeds the VoI.

## What this gap is not

- It is not "add a read penalty" (reward shaping). Every arm has the same reward.
- It is not "verify more". The optimum skips reads where they carry no information.
- It is not a restatement of the benchmark. It is a claim about an estimator, which can be falsified in a tabular
  model and in the LLM agent.
