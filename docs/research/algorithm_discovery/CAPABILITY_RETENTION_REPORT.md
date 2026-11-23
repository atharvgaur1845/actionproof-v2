# Capability retention report

**Harness.** The main session's R1 harness, reused unchanged as instructed: frozen inputs in
`data/retention/r1_inputs.jsonl`, `actionproof/retention/gen.py` (greedy, merged checkpoints, the 8,192-token
context respected), and `scripts/r1_score.py` in `.venv-r1` with the official checkers:
- IFEval: lm-eval 0.4.13, prompt- and instruction-level, strict and loose;
- BFCL v4 single-turn: AST categories and irrelevance.

**Models.** Seed 0 of grpo and of each candidate that entered C1. R1 already scored the warm start and the base
model.

**Main session's finding** (reported to this programme on 2026-11-23, not recomputed here): no measurable capability
loss for any earlier arm (R1 H2 non-inferior).

**Question for this programme.** Do the costed-verification estimators preserve IFEval and BFCL relative to the warm
start, as earlier ActionProof RL did?

## Results

PENDING: filled from `results/r1/score/ad_*.summary.json` only.
