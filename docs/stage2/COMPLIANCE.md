# Stage 2 compliance checklist

Sources: hackathon.lexsi.ai, its kickoff slides and submission page (read 2026-11-20); the 6+6 paper format was given to finalists privately (6 main pages + up to 6 appendix pages, references not counted). Final submission by Nov 24 23:59 AoE = **Nov 25 17:29 IST**; we submit by Nov 25 12:00 IST. The latest submission counts. Each item links to the artefact that satisfies it; an item with no artefact is not done.

## Required by the write-up template

| Item | Artefact | Status |
|---|---|---|
| Track and problem-statement number | Track 3, PS9 (README) | done |
| Title and one-line claim | `docs/paper/DRAFT.md` (title fixed by the registered A14 outcome) | drafted |
| Headline before → after number (same model) | `claims/claims.yaml` C1-H2 (warm start vs R_V; proc_conf) | to write into the paper |
| Base model and method in one line | paper §3 (Tiny Aya Global 3.35B, LoRA warm start, GRPO with R_V vs R_A) | drafted |
| Why this problem / real use case | paper §1, `docs/stage2/PRODUCT.md` | drafted |
| Abstract under 500 words | paper | not written |
| Problem statement and approach | paper §1–3 | drafted |
| Hackathon repos used and how they connect (AgentTune, AuditKIT) | README "Toolkit chain"; paper appendix | Stage 1 done; Stage 2 to state |
| Models and licences | README; paper appendix | to state for Stage 2 |
| Datasets and benchmarks, splits, leak checks | `docs/stage2/BENCHMARK_CARD.md`, `tests/integrity/test_procsim.py` | done |
| Metrics and why | `docs/stage2/METRICS.md`, paper §3 | done |
| Method, training setup, hyperparameters | `PREREGISTRATION_A12.md` §5, paper §3 | done |
| Results: same model before vs after | paper §4 table | to build |
| Limitations | paper §6 | drafted |
| **Declaration of work done outside the hackathon** (required): Stage 1 (Nov 1–15) vs Stage 2 (Nov 20–25) | paper appendix, README | not written |
| "With more credits": planned approach and expected before/after | paper §6 | not written |
| Artefacts: HF model and dataset collections, repositories | README (Stage 1); Stage 2 adapters on the private dataset | Stage 2 publishing needs the user's approval |
| Repo layout and run instructions | README, `docs/stage2/COMMANDS.md` | update for Stage 2 |
| Teammates tagged | — | user |

## Recommended

| Item | Artefact | Status |
|---|---|---|
| Seeds, each reported separately | `results/s2/a12_summary.json` (5 seeds), `results/a14/a14_summary.json` | done |
| Ablations | A14 interface ablation, A11 four-mode factorial, reward-signal analysis | done |
| Confidence intervals and significance tests | `scripts/s2_stats.py`, `scripts/a14_stats.py` | done |
| Negative results | H3 failed; A14 thesis abandoned; A11 interaction not supported; A13 Fisher test invalid (D-A13-1) | done; in `docs/stage2/DECISIONS.md` |
| Compute used, with cost | `docs/stage2/HF_CREDITS.md` | to total |
| LLM usage statement | paper appendix | not written |
| Author contributions | paper | user |

## Track 3 write-up metrics (agentic trajectories)

| Item | Artefact | Status |
|---|---|---|
| Task success (verified completion next to claimed success), per subset | confirmatory summaries `results/eval/a12/conf/*.summary.json` (VSC, false success, per fault class) | done |
| Tool-call accuracy (validity), per subset | same summaries (`tool_call_validity`) | done |
| Average latency, per subset | `results/s2/latency_*.json` (batch-1, p50/p95 our addition); gateway overhead `results/s2/latency_gateway.json` | running |

## Optional (general)

| Item | Artefact | Status |
|---|---|---|
| Judge–human agreement (% or κ), with labelling protocol | `scripts/h4_analyze.py`, `PREREGISTRATION_A11.md` §9 | waiting for the user's 60 labels; one annotator, not consensus |

## Our own additions

| Item | Artefact | Status |
|---|---|---|
| Benchmark card | `docs/stage2/BENCHMARK_CARD.md` | done |
| Claim-to-evidence ledger | `claims/claims.yaml`, `scripts/s2_numcheck.py` (README, video script, product page) | done; add the paper when final |
| Every `freeze_*.py --check` passes, or its failure is a logged deviation | `scripts/ci_freeze_check.py` in CI | done |
| Public demo or repository link | GitHub repository; dashboard via `actionproof serve` | needs the user's approval to make public artefacts |
| Anonymity for a later double-blind venue | rename or make private the public HF models with "actionproof" in the name | later, not for this submission |
