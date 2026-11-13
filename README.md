# ActionProof

**Agents say "done". We check the database. Then we ask what RL taught them.**

ActionProof is an environment, a set of rewards and a controlled training study for tool-using agents whose tool acknowledgements can lie: phantom, partial, stale, timed-out and ambiguous writes. Every number is checked against the environment's database and receipt log, never against the transcript.

**Headline.** With the warm start and reward form held fixed, a reward whose goal term reads the database makes GRPO amplify post-write verification (sampled 0.25 → 0.72). The same reward computed from the tool's acknowledgements removes verification (→ 0.10). What gets selected is a script: on a tool outside training the agent reads the contradicting evidence and still reports "done". See `docs/ABSTRACT.md`.

- **Track:** 3, Agentic Fine-tuning.
- **Toolkit chain:** AgentTune → AuditKIT, plus CuratorKIT.
- **Models:** Cohere Tiny Aya Global (policy) and Aya Expanse 8B / 32B (teacher and judges).
- **Write-up:** `docs/ARTICLE_DRAFT.md` (abstract: `docs/ABSTRACT.md`).
- **Novelty and related work:** `docs/research/novelty_scouts_2026-11-11.md`.
- **Live status and next actions:** `docs/NEXT_ACTIONS.md`.
- **Latest amendment:** `PREREGISTRATION_A8.md` (X1: does the acknowledgement reward remove refund read-back only through refund-timeout and unrecoverable-refund tasks? Audit `results/a8/ack_class_audit.md`; run pending). Before it, `PREREGISTRATION_A7.md` (results in `results/eval/a6/` and `results/eval/a7/`, corrected forced-prefix battery `results/a7/battery_v2.md`). Every deviation is logged in `docs/SPEC.md` §27.
- **Pre-registration:** `PREREGISTRATION.md`.
- **Full spec and deviation log:** `docs/SPEC.md`.

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
for f in freeze_bench freeze_a4 freeze_a5 freeze_a6 freeze_a7 freeze_a8; do .venv/bin/python scripts/$f.py --check; done
```

The tests cover:
- fault semantics;
- the PCSTR ordering property (property-based);
- the reward sanity matrix;
- an oracle that solves 100% of every split;
- deterministic replay;
- the banking domain and the language variants.

Every check must print `OK`, except `freeze_a5`, which reports `verifier_a5.py` and `a5_rewards.py`. That mismatch is the logged deviation D-A5-3 (a parsing fix after registration; SPEC §27, 2026-11-10). It was not re-registered on purpose. Checked from a fresh clone of `706af05` on 2026-11-13: 146 tests passed, 4 skipped.

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

- Code: Apache-2.0.
- Models and adapters derived from Cohere Aya are CC-BY-NC-4.0, with Cohere attribution.
