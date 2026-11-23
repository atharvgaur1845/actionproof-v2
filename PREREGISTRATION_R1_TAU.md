# PREREGISTRATION R1-T: τ-bench retail retention check (addendum to R1)

Registered 2026-11-23 before any τ-bench output; time in `results/r1_tau_hashes.json`. Selection reasons: `docs/upgrade/PLAN.md` §2.

## 1. Question
Does ProcureSim training change task success in a different database-changing agent setting graded by final database state?
**Descriptive only.** With 115 tasks and a likely floor-level 3B score, the study is powered for large changes only; no hypothesis test is registered.

## 2. Models
`base`, `warm_start`, `a12_rv_s0`–`s2`, `rsc_s0`–`s2` (as in PREREGISTRATION_R1.md §2; seeds 0–2 only, chosen now for compute).

## 3. Protocol
- τ-bench `sierra-research/tau-bench` @ 59a200c (MIT), retail, **test split (115 tasks)**, 3 trials per task, unchanged environment, tasks, reward (database state + required outputs) and `ChatReActAgent` (ReAct text format, temperature 0, 30 steps) — the only τ-bench agent usable without native function calling.
- `litellm` replaced by `actionproof/retention/tau_shim/litellm.py` (requests forwarded unchanged to local vLLM servers; cost accounting dropped).
- Simulated user: **Aya Expanse 32B** (CC-BY-NC-4.0, research use) on vLLM with server-default sampling (vLLM defaults, temperature 1.0, as the OpenAI default the paper's GPT-4o user had), identical for every model; it never grades. Not the paper's user model, so **absolute numbers are not comparable with published τ-bench results**.
- Agent: the checkpoint under test, 8,192-token context, at most 512 new tokens per step (the remaining room if less).
- Context overflow ends the episode with reward 0, flagged `overflow`; other exceptions with reward 0, flagged `error`. Both rates are reported per model.

## 4. Feasibility pilot (decides whether the main run happens)
`base`, first 20 test tasks, 1 trial, written to `results/r1_tau/pilot/` and not analysed otherwise. **If the base model overflows in more than 50% of pilot episodes, the main run is cancelled** and τ-bench is reported as infeasible for an 8K-context 3B model under this protocol, with the pilot numbers. Otherwise tasks 1105–1106 run once.

## 5. Analysis
`scripts/r1_tau_stats.py` (frozen with this file): per model pass^1, pass^k (k = 1..3), overflow and error rates, mean messages; contrasts warm − base, a12_rv − warm, rsc − warm on per-task pass^1 with a paired task bootstrap. All descriptive.
