# Generalisation report

Scope, registered in PREREGISTRATION_AD_C1.md §4. Everything here is descriptive: seed 0 of each C1 arm.

| Shift | Split | Environment | What it tests |
|---|---|---|---|
| Deployment split | `cw_conf` (300, locked) | CW, c = 0.05 | Held-out tasks from the training prior (confirmatory) |
| Free-read standard tasks | `proc_dev` (150) | Stage 3, dense faults, no cost in the metric | Does costed training damage ActionProof's standard reliability? |
| Renamed tools | `s4_api` | S4 API-shifted environment (`scripts/s4_eval.py`) | Does the learned verification survive a new tool surface? |
| Language | `s4_lang_hi` | Hindi requests | Language transfer of verification |
| Unseen workflow | `s4_wf4` | 4-step workflow | Workflow transfer |
| Cost shift | c = 0.30 (D1 tasks 1209, 1211) | CW | Does the estimator stop verifying when verification no longer pays? |

**Not evaluated, and why:**
- τ-bench, BFCL multi-turn and AgentBench as *reliability* benchmarks. None has silent write failures or a read cost.
  The main session's R1 covers BFCL single-turn and IFEval for capability retention (CAPABILITY_RETENTION_REPORT.md).
- Other model families: one base model only.

## Results

PENDING: filled from `results/ad/conf/`, `results/ad/shift/` and `results/ad/dev/*proc_dev*` only.
