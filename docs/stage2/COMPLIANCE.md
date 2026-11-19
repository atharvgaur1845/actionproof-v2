# Stage 2 compliance checklist

Source: hackathon.lexsi.ai, read 2026-11-20 02:10 IST. Final submission: a video presentation or slides by Nov 24 23:59 AoE (Nov 25 17:29 IST). No length limit is stated. Each item links to the artefact that satisfies it; an item with no link is not done.

## Write-up template (required items marked on the page)

| Item | Artefact | Status |
|---|---|---|
| Title and one-line claim | | |
| Abstract under 500 words | | |
| Problem statement and approach | | |
| Hackathon repos used and how they connect (AgentTune for post-training, AuditKIT for evaluation) | | |
| Models and licences (base Aya model) | | |
| Datasets, benchmarks and licences; train/test splits; leak check | | |
| Metrics, and why these | | |
| Method and experiments: training setup, hyperparameters | | |
| Results: same model before vs after (table) | | |
| Limitations | | |
| Artefacts: HF model and dataset collections, repositories | | |
| Repo layout and run instructions | | |
| Teammates tagged | | |

## Recommended items

| Item | Artefact | Status |
|---|---|---|
| Seeds (each seed reported separately) | | |
| Ablations | | |
| Confidence intervals and significance tests | | |
| Negative results (what did not work) | | |
| Compute used, with cost estimates | | |
| Declaration of work done outside the hackathon, and of what existed before Stage 2 (Nov 1–15) vs what Stage 2 produced (Nov 20–25) | | |
| LLM usage statement | | |
| Author contributions | | |

## Track 3 metrics

| Item | Artefact | Status |
|---|---|---|
| Task success (verified completion next to claimed success), per subset | | |
| Tool-call accuracy (validity and argument correctness), per subset | | |
| Average latency (plus p50/p95), per subset | | |
| LLM-judge agreement with human labels, with the labelling protocol | | |

## Our own additions

| Item | Artefact | Status |
|---|---|---|
| Benchmark card | `docs/stage2/BENCHMARK_CARD.md` | skeleton |
| Claim-to-evidence ledger resolving every published number | `claims/claims.yaml` | skeleton |
| Every `freeze_*.py --check` passes, or its failure is a logged deviation | | |
| Public demo or repository link | | |
