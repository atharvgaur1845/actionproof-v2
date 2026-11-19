# Stage 2 risk register

Likelihood and impact: L (low), M (medium), H (high). Owner is Claude unless stated.

| # | Risk | L | I | Mitigation | Trigger and fallback |
|---|---|---|---|---|---|
| R1 | ProcureSim is not ready for GRPO by Sat Nov 21 14:00 | M | H | Spec first (`VERIFIER_SPEC.md`); oracle and acknowledgement tests before anything else | Gate fails → procurement becomes evaluation-only; C1 reported as "not replicated in time" |
| R2 | The warm start is outside the learnable band (gate fails) | M | H | Mixed demonstrators (30% verifying), predicted non-zero-variance share checked before training | One re-mix; a second failure is reported as a negative |
| R3 | H200 capacity: one submitted job per user, 2 GPUs | H | H | Two-lane worker; A100 worker; HF Jobs only after a full-path smoke test | **User rule: any job pending more than 3–4 h moves to HF Jobs** (logged) |
| R4 | The co-user's pending A100/H100 jobs take our shared QOS slots | M | M | Never touch their jobs; prefer H200 for critical runs | H1 not started by Fri 20:00 → HF A100 |
| R5 | A wrong verifier or recovery decision (bad projection, retry into a duplicate) | M | H | Independent reference verifier, independent golden traces, mutation tests, model check, fail-closed defaults | Any golden failure blocks the confirmatory run until fixed and logged |
| R6 | Confirmatory splits leak through checkpoint selection | L | H | Step-200 default; dev/val only for intermediate checkpoints; frozen hashes; access log | Any second run on a confirmatory split is a logged deviation |
| R7 | Too much work for 5 days | H | H | Effort split 45/30/15/10; cut order in `PLAN.md` §H.1; no new features | Cut order applied at each milestone review |
| R8 | Parallel agents add integration risk | M | M | Worktrees; merge only behind tests; one workstream integrated at a time | Revert the merge |
| R9 | Live Aya fails on camera | M | M | Labelled scripted fallback; vLLM tunnel; recorded segments | Use the fallback and say so on screen |
| R10 | HF spend overruns | L | M | Caps: jobs $150, endpoint $40, hard stop $195; log every job | Stop new HF jobs at the cap |
| R11 | Overlap with LIMBO, Cordon or 2611.23353 is judged as lack of novelty | M | H | Narrow research claim (C1); gateway not claimed as novel | — |
| R12 | Frozen files edited by mistake | L | H | CI frozen-file guard; `freeze_*.py --check` at every milestone | Revert and log |
