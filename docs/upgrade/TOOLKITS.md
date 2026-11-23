# Hackathon toolkit usage (deliverable K / Mission 9)

Every row is checked against the code by `grep` for the import and the call site, on branch `actionproof-research-upgrade` (2026-11-23). Vendored revisions are on the cluster under `vendor/`.

| Kit | Revision | Used? | Where, and what it contributed |
|---|---|---|---|
| **AgentTune** | d4db4c1 (2026-10-07) | **Yes: every GRPO run** | `TrlAgenticGrpo` is the trainer behind every Stage 2–5 policy: `actionproof/stage3/train.py`, `stage4/train_c5g.py`, `stage4/train_fca.py`, `train/grpo_proc.py`, `train/dpo.py`. The tool-call parser (`extract_tool_calls_from_text`) and the tools prompt are used in rollouts and evaluation (`actionproof/rollout/agenttune_rollout.py`, `rollout/episode.py`), so the policy is trained and evaluated in one format. `create_agentic_trainer` is used in Stage 1. Our batched vLLM rollout (`actionproof/rollout/batched_vllm.py`) replaces AgentTune's one-at-a-time Aya tool loop for speed (G0: 415 s for 8 rollouts), keeping AgentTune's token-mask convention |
| **AuditKIT** | 71316da (2026-10-07) | **Yes: Stage 1 judge audit** | `TaskCompletion` metric and `AgentEvalRunner` / `FinalStateAssertion` (`actionproof/eval/auditkit_bridge.py`, `eval/judge_audit.py`, `scripts/judge_instructed.py`). Result: with the receipt ledger shown, an Aya Expanse 32B judge using AuditKIT's TaskCompletion prompt still accepts 71% (213 of 298) of false completions (README). The J1 committee extends this audit with four independent families and human calibration |
| **CuratorKIT** | dd9e835 (2026-10-12) | **Yes: Stage 1 data** | `ExactDeduplicator`, `MinHashDeduplicator` and `DataSample` deduplicate the naturalistic acknowledgement texts (`actionproof/bench/naturalistic_acks.py`) |
| **CircuitKIT** | 0.2.4 | **Attempted, failed** | `scripts/a16_circuitkit.py` (EAP-IG, exploratory). Deviation D-A16-1: discovery fails importing a transformers class in the job environment. No CircuitKIT result is claimed; A16's ablation is not specific (`docs/final/FINAL_CLAIMS.md`: never say "circuit") |
| **AlignTune** | — | **Not used** | No import anywhere in the repository |
| **SafeTune** | — | **Not used** | No import anywhere in the repository |

## Pipeline

1. CuratorKIT deduplicates the acknowledgement texts (Stage 1).
2. The ProcureSim and SupportSim tasks are frozen and hashed.
3. AgentTune GRPO, with our batched rollout and the R_V, R_A or PCSTR rewards, trains each policy.
4. Our verifier scores the policies (the deterministic SQLite truth).
5. AuditKIT's judge audit and the J1 committee measure what LLM judges miss.
6. R1 and R1-T measure retention with the official BFCL, IFEval and τ-bench code.
