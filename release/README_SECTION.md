<!--
Draft section for README.md (paste after "Deliverables"). It contains no result numbers on purpose: README.md is scanned
by scripts/s2_numcheck.py, and every result-like number in it must resolve through claims/claims.yaml.
Placeholders to fill before pasting: <LIVE_SPACE_URL>. The paper PDF is built from paper/main_v3.tex and is not tracked in git.
-->

## Public release

| What | Link |
|---|---|
| Live Space (the recommended agent, running) | <LIVE_SPACE_URL> |
| Replay Space (recorded episodes, no model calls) | https://huggingface.co/spaces/atharvgaur18/actionproof |
| Recommended agent D (merged, seed 0) | https://huggingface.co/atharvgaur18/actionproof-agent-d |
| D adapters (seeds 0 to 2) and the Dw adapter | https://huggingface.co/atharvgaur18/actionproof-agent-d-adapters |
| Recovery-SFT warm start Dw | https://huggingface.co/atharvgaur18/actionproof-warmstart-dw |
| Reference agent A (recovery-start GRPO) | https://huggingface.co/atharvgaur18/actionproof-agent-a |
| Dn, the no-std GRPO null result (P3), for reproducibility | https://huggingface.co/atharvgaur18/actionproof-agent-dn |
| Episode traces, frozen splits, judge cases | https://huggingface.co/datasets/atharvgaur18/actionproof-traces |
| Paper | `paper/main_v3.pdf` (source `paper/main_v3.tex`) |
| Write-up | `docs/submission/WRITEUP.md` |
| Code | https://github.com/atharvgaur1845/actionproof-v2 |

All models are LoRA fine-tunes of `CohereLabs/tiny-aya-global` and carry CC-BY-NC-4.0: research use only, no production
payments. Every result is on a simulator (ProcureSim). Arms C and D are not nested, P3 is a null result, and the
agents do not generalise to renamed tools or to the four-step workflow; each model card says so next to its numbers.

### Verify the numbers without a GPU (about 10 minutes)

```bash
git clone https://github.com/atharvgaur1845/actionproof-v2 && cd actionproof-v2
pip install numpy pyyaml huggingface_hub
bash scripts/verify_numbers.sh
```

The script:
1. downloads the episode files (`results/{p1,p2,p3}/**/*.jsonl`) from the traces dataset into `results/`;
2. reruns `scripts/p1_stats.py`, `scripts/p2_stats.py`, `scripts/p3_stats.py` and `scripts/audit/dwarm_2x2.py`
   (fixed bootstrap seeds, so the output is deterministic);
3. requires `git diff --exit-code results/` to be empty, that is, the recomputed summaries are byte-identical to the
   committed ones;
4. runs `python scripts/s2_numcheck.py`, which checks every number in the paper, write-up and README against
   `claims/claims.yaml`.

Most of the time is the download (a couple of GB); the analyses take a few minutes on one CPU core.

### Toolkits actually used

Checked against the code by import, not by intent.

| Toolkit | What it does in the pipeline | Where |
|---|---|---|
| CuratorKIT | Exact and MinHash deduplication of the oracle recovery demonstrations before SFT (and of acknowledgement texts in Stage 1) | `actionproof/p1/build.py`, `actionproof/bench/naturalistic_acks.py` |
| AgentTune | The tool-call format: system-prompt block, schema builder and `extract_tool_calls_from_text` parser, shared by training and evaluation; `TrlAgenticGrpo` trains every GRPO agent (A, D, Dn) | `actionproof.rollout.episode._agenttune`, `scripts/p1_eval.py`, `actionproof/stage3/train.py`, `actionproof/p3/train_nostd.py` |
| Deterministic verifier (ours) | Scores every episode against the database; the authority for all results | `actionproof/integrity/sim/truth.py` |
| AuditKIT | `TaskCompletion` LLM-judge scores of the P1 episodes, secondary and descriptive; it never decides an outcome | `scripts/p1_auditkit.py`, `actionproof/eval/judge_audit.py` |

Not used for any result: AlignTune and SafeTune (never imported). CircuitKIT was attempted in the exploratory A16
analysis, failed in the job environment, and no result is claimed from it (`docs/upgrade/TOOLKITS.md`).
The recovery SFT warm starts (W, Dw) are plain TRL/PEFT fine-tuning (`actionproof/train/rft.py`), not a toolkit.
