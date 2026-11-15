# ActionProof

**Agents say "done". We check the database. Then we ask what RL taught them.**

ActionProof is an environment, a set of rewards and a controlled training study for tool-using agents whose tool acknowledgements can lie: phantom, partial, stale, timed-out and ambiguous writes. Every number is checked against the environment's database and receipt log, never against the transcript.

**PS9 in one paragraph.** Rewards that read a tool's acknowledgements or the transcript get gamed: a tool-trusting GRPO reward leaves verified success at 0.635–0.645, and AuditKIT's own TaskCompletion judge (Aya Expanse 32B) accepts 95.2% of false "completed" claims; handed the database's receipt ledger, it still accepts 71.5% of failures the ledger itself exposes (pre-registered, `results/judge/grounded_audit.md`). A reward that reads the database (PCSTR-d) raises verified safe completion from 0.6425 to **0.8425** on 400 untouched pre-registered tasks, **+20.25 pp** [16.4, 24.3] over the tool-trusting reward, and the report text cannot move its state terms. The mechanism is traced to two task classes, where checking forfeits acknowledgement reward: removing them keeps refund read-back at 0.669 while a 60-random-task placebo stays at 0.000 (pre-registered, one seed per cell).

## Deliverables

| PS9 deliverable | Where |
|---|---|
| Interactive write-up (website) | https://atharvgaur18-actionproof.static.hf.space/index.html (Space: https://huggingface.co/spaces/atharvgaur18/actionproof) |
| Code (this repository) | https://github.com/atharvgaur1845/actionproof-v2 |
| HF models collection (29 LoRA adapters) | https://huggingface.co/collections/atharvgaur18/actionproof-adapters-6af7fb12592ac424c2086c04 |
| HF datasets collection (benchmark, tool-call logs and traces) | https://huggingface.co/collections/atharvgaur18/actionproof-benchmark-traces-demo-6af7fb25037f3d7eaf294e22 |
| Write-up: metrics, methods, results, limitations | `docs/ARTICLE_DRAFT.md` (abstract `docs/ABSTRACT.md`) |
| Agentic trajectory metrics (task success, tool-call accuracy, average latency) | `results/agentic_metrics.md` (`scripts/agentic_metrics.py`); latency there is batched seconds per episode; single-request p50 / p95 per episode in `results/latency_single.md` (A100, batch 1) |
| Reproduction | Section "Reproduce" below |

- **Track:** 3, Agentic Fine-tuning.
- **Toolkit chain:** AgentTune → AuditKIT, plus CuratorKIT.
- **Models:** Cohere Tiny Aya Global (policy) and Aya Expanse 8B / 32B (teacher and judges).
- **Novelty and related work:** `docs/research/novelty_scouts_2026-11-11.md`.
- **Environment v2** (`PREREGISTRATION_A10_ADDENDUM.md`, `results/a10/v2_env.md`): scoped idempotency and timeouts before the commit. Existing results unchanged on the same tasks, but the state-reward policies fail on timeouts that did not commit (PCSTR-d 0.176 vs tool_trust 0.706): they read, then report completed without retrying.
- **Latest amendments:** `PREREGISTRATION_A10.md` (strict metric SVC: state arms SVC = VSC, PCSTR-d − tool_trust +41.9 pp on SVC; dependency-aware reward, offline: H-A10c holds, H-A10d fails, `results/a10/`), `PREREGISTRATION_A9.md` (grounded judge audit, `results/judge/grounded_audit.md`; task 421 not run), and an independent reference verifier agreeing on 4,000 / 4,000 confirmatory episodes (`results/reference_verifier.md`). Before them, `PREREGISTRATION_A8.md` and its addenda. X1 and the placebo are **completed** (`results/a8/x1_readback.md`); seed 1 of the A7 state arm (task 415) reproduces seed 0. Addendum 2 (`PREREGISTRATION_A8_ADDENDUM2.md`, registered 2026-11-14 before either run): task 417 (state reward on the X1 pool) and task 418 (A7 ack reward, seed 1); both finished: the state reward on the X1 pool reads back 152/152, interaction +0.678 [0.605, 0.750] (`results/a8/x1_interaction.md`); the ack arm's seed 1 reads back 0/162 like seed 0. All three registered predictions met. Before it, `PREREGISTRATION_A7.md` (results in `results/eval/a6/`, `results/eval/a7/`, corrected forced-prefix battery `results/a7/battery_v2.md`). Every deviation is logged in `docs/SPEC.md` §27.
- **Pre-registration:** `PREREGISTRATION.md`. **Full spec and deviation log:** `docs/SPEC.md`.

## Repository map

| Path | What |
|---|---|
| `actionproof/env/` | SQLite environments (e-commerce `tools_ecom.py`, banking `tools_bank.py`), fault schedule, receipt ledger, verifiers, counterfactual footprint |
| `actionproof/bench/` | Task generators (all splits), prompts, HI/Hinglish/pressure variants, naturalistic F8 acks |
| `actionproof/rewards/` | PCSTR / PCSTR-d, baselines (`tool_trust`, `outcome`), LLM judge, scripted agents (oracle, lean expert) |
| `actionproof/rollout/` | AgentTune Cohere-format episodes, batched multi-turn vLLM rollouts |
| `actionproof/train/` | RFT data and training, DPO, GRPO (AgentTune `TrlAgenticGrpo`), curriculum, expert demos |
| `actionproof/eval/` | Batched evaluation, stats, per-fault breakdown, calibration, footprint, AuditKIT bridge, judge audit, ROI |
| `actionproof/interp/` | Activation extraction, linear probes, text baseline, probe guard, knowing-liar analysis |
| `slurm/` | Cluster worker queue (`worker.sh`, `tasks/*.sh`), env setup, HF Jobs scripts |
| `demo/` | Gradio Space (live ZeroGPU runs and replays of real evaluation episodes) |
| `scripts/freeze_bench.py` | Writes `data/bench/*.jsonl` and checks every split and scoring-code hash against the pre-registration |

## Reproduce

### 1. Environment

Python 3.12, CUDA 13 driver (≥ 580). Toolkits are pinned to these commits: AgentTune `d4db4c12d523`, AuditKIT `71316da34578`, CuratorKIT `dd9e835301fc`.

```bash
bash slurm/setup_env.sh
```

This clones the toolkits at those SHAs, creates `.venv` with uv and installs everything. `requirements.lock` holds the exact resolved versions: torch 2.11.0+cu130, transformers 5.22.1, trl 1.7.1, vLLM 0.23.0, peft 0.21.2.

### 2. Tests and benchmark integrity (CPU only)

```bash
.venv/bin/python -m pytest -q tests/
for f in freeze_bench freeze_a4 freeze_a5 freeze_a6 freeze_a7 freeze_a8 freeze_a8b freeze_a8c freeze_a9 freeze_a10; do .venv/bin/python scripts/$f.py --check; done
```

The tests cover:
- fault semantics;
- the PCSTR ordering property (property-based);
- the reward sanity matrix;
- an oracle that solves 100% of every split;
- deterministic replay;
- the banking domain and the language variants.

Every check must print `OK`, except `freeze_a9` (D-A9-1, `judge_audit.py`) and `freeze_a5`, which reports `verifier_a5.py` and `a5_rewards.py`. That mismatch is the logged deviation D-A5-3 (a parsing fix after registration; SPEC §27, 2026-11-10). It was not re-registered on purpose. Checked from a fresh GitHub clone of `6bb67b4` on 2026-11-15 12:10 IST (tests run with the project's existing `.venv`, not a fresh `setup_env.sh`): 506 tests passed, 4 skipped; all nine freeze checks OK except `freeze_a5` (D-A5-3) and `freeze_a9`, which reports `judge_audit.py` (D-A9-1, a one-line FP8 fix after registration; SPEC §27). Later commits change documents, result files and the rebuilt site only. Results on the website and in the paper are the files in this repository; the site footer names the commit it was built from.

### 3. Pipeline (each step is a queue task in `slurm/tasks/`)

| Step | Task | Command core |
|---|---|---|
| Teacher trajectories (Aya 32B, verify prompt) | `070` | `actionproof.eval.run_eval --model CohereLabs/aya-expanse-32b --prompt verify_a --split train --samples 8` |
| RFT v2 data + training | `087` | `actionproof.train.expert_demos` → `actionproof.train.rft_data --expert …` → `actionproof.train.rft` |
| Learnable-band curriculum | `091` | `run_eval --samples 8 --temperature 1.0` → `actionproof.train.curriculum` |
| GRPO arms (shared init, only the reward differs) | `150–157` | `slurm/grpo_arm.sh <arm> runs/rft_tiny_v2 data/curriculum/band_tiny_v2.jsonl v2` |
| Leaderboard | `300–302` | `run_eval` over 7 Cohere models × prompts × splits → `actionproof.eval.leaderboard` |
| Judge-fooling audit | `170` | `actionproof.eval.judge_audit build/judge` |
| ROI | `175` | timed `run_eval` → `actionproof.eval.roi` |

Run tasks on a SLURM cluster with `sbatch slurm/worker.sh <pool>`. Workers claim tasks from `queue/<pool>/pending` in order, respect `# after:` dependencies, recover tasks stranded by preemption, and pause after 2 consecutive failures.

If no cluster GPU is free, the same commands run on HF Jobs: `slurm/hfjob_rft_v2.sh` and `slurm/hfjob_curriculum_v2.sh` reproduce the exact `requirements.lock` environment.

### 4. Analysis

```bash
.venv/bin/python -m actionproof.eval.by_fault results/eval/r3/<run>__val.jsonl        # VSC / FCR / post-ack check per fault
.venv/bin/python -m actionproof.eval.calibration results/eval/r3/<run>__test_iid.jsonl  # per-service verification vs fault rate
.venv/bin/python -m actionproof.eval.footprint_report results/eval/r3/<run>__val.jsonl  # unnecessary side effects, NWCG
.venv/bin/python -m actionproof.eval.auditkit_bridge results/eval/r3/<run>__test_iid.jsonl  # AuditKIT verified outcomes
```

## Determinism

- Greedy evaluation of a saved adapter is deterministic. The env seed depends only on the task.
- Sampled evaluation uses a reproducible per-(task, sample, turn) seed (see PREREGISTRATION, Amendment 3).

## License

- Code: Apache-2.0 (`LICENSE`).
- Models and adapters derived from Cohere Aya are CC-BY-NC-4.0, with Cohere attribution.
