# ActionProof

**Agents say "done". We check the database. Then we ask what RL taught them.**

ActionProof is an environment, a set of rewards and a controlled training study for tool-using agents whose tool acknowledgements can lie: phantom, partial, stale, timed-out and ambiguous writes. Every number is checked against the environment's database and receipt log, never against the transcript.

**Status (2026-11-21).** How the labels map:
- **Hackathon rounds.** "Stage 1" (Nov 1–15) is the preliminary round and "Stage 2" (Nov 20–25) the finalist round.
- **Internal research phases.** "Stage 3" (`docs/stage3/`, `actionproof/stage3/`) and "Stage 4" (`docs/research/`, `actionproof/stage4/`, branch `stage4`) are internal phases run inside the Stage 2 window. They are not hackathon rounds.

Every number resolves through `claims/claims.yaml` (`python scripts/s2_numcheck.py`). Contradictions found and fixed: `docs/CONSISTENCY_AUDIT.md`.

Stage 2 results (finalist round):
- **A12.** Training with a state reward (R_V) made read-back after phantom writes the default (1.00), while an acknowledgement reward (R_A) of identical form left the warm start's behaviour unchanged (0.00), in 5 of 5 seeds from one warm start. R_A's greedy behaviour equals the untrained warm start's.
- **ID1 (registered, 2 seeds).** From a warm start whose demonstrations never read, R_V never samples a read in 200 steps and read-back stays 0/153: the reward selects checking from the starting model's repertoire; it does not create it.
- **A14, A15.** Given forced evidence, every model handles present and absent payments. Partial-payment repair appears almost only under R_V. It generalises across ratios, but only within the pay-first workflow: about 0% after a goods receipt, and 5–55% after an order (seeds 0–2).
- **A16.** All models encode partial vs complete. Training changes the action margin. The ablation that lowers it is not specific to R_V.
- **A17.** In healthcare administration, R_V reads back more than R_A, mainly because R_A reads less than the warm start. R_V vs warm start is not established, and IT operations shows no difference.
- **K1.** Training-pool composition explains part of R_A's drift away from reading, not its zero read-back.

Stage 3 (`PREREGISTRATION_S3.md`; corrected by `PREREGISTRATION_S3_ADDENDUM1.md`, because the pool also added fault types):
- **Against A12 R_V (a recipe comparison).** The recovery-start training recipe (RSC) raises VSC by +0.21 on fresh recovery and full-episode splits. Every one of its 5 runs beats every A12 R_V run. Of the full-episode gain, 0.12 is duplicate delivery, a fault type A12 never trained on.
- **Against a class-matched control (the start-state comparison).** With five runs per arm (two control seeds added by a registered extension, `PREREGISTRATION_S3_ADDENDUM2.md`), recovery starts improve completion from faulted states across training runs (+0.107, unpaired seed permutation p 0.012). The partial-payment gain on full episodes does not reach the registered rule (p 0.056): partial repair was learned in 5 of 5 RSC runs and 2 of 5 control runs. The three-seed safety trade-off reversed with five control seeds (violations -0.021).
- **Still unsolved.** Partial payments after a goods receipt are repaired by no arm (RSC 5 of 60 episodes). Raising them from 6 to 21 of 400 training tasks did not help (0.005 on 100 targeted tasks; registered, 2 runs), so pool frequency is not the explanation.
- **Transfer is negative.** In healthcare administration, VSC falls from 0.158 (warm start) to 0.080 (A12 R_V) to 0.047 (RSC).

Stage 4 (`docs/research/`):
- **Shifts** (`PREREGISTRATION_S4_EVAL.md` and its addendum). The RSC gain survives Hindi (+0.13), Hinglish (+0.07) and two-fault episodes (+0.07), each in 5 of 5 seeds. It vanishes under renamed tools and in an unseen four-step workflow, where every model runs the trained three-step sequence and skips the new step.
- **No new algorithm.** The novelty audit found every candidate published. FCA, a forked counterfactual credit estimator, failed its pre-set development criterion and was dropped (`docs/research/algorithm/FCA_RESULT.md`).
- **Addendum 4 (registered): inconclusive.** Recovery starts that end at the acknowledgement, without the handed read, learn partial repair in 4 of 5 runs (control 2 of 5, recovery starts 5 of 5). Neither contrast meets the registered rule (Holm 0.206 and 0.262), so whether the handed evidence matters is not settled.
- **Running:** SR1 (tool-name randomisation), evaluation in progress.

**Product (`product/`, branch `stage4`).** Run it locally with `python -m product.app` (port 7860). It has not been deployed.
- **Live, deterministic and CPU-only** (the real ProcureSim simulator and verifier; no model calls):
  - *Incident simulator.* Choose a workflow, a fault, the faulted step, the partial ratio and the agent. It shows an execution-truth timeline that separates what the agent saw, what the database did, and the evaluator-only verdict.
  - *Reward lab.* The same trajectory scored by R_V, R_A, the exploratory safety-first reward and VSC.
  - *Recovery arena.* Step through a faulted workflow, choosing read, retry with the same or a new key, compensate, continue, escalate or stop.
- **Recorded:** the policy comparison replays real greedy evaluation episodes (warm start, R_A, R_V, class-matched control, RSC) on the same fresh task. It is not live inference.
- **Generated:** the research dashboard is built from result files by `python -m product.build_data`.

**In one paragraph (Stage 1, Nov 1–15).** When tool acknowledgements can contradict execution state, does the information source of a reinforcement-learning reward change whether an agent checks its own writes? **Registered confirmatory result** (400 untouched tasks, `PREREGISTRATION.md` H1): PCSTR-d (state-grounded, two training seeds) reaches verified-success completion (VSC) 0.8425 against 0.8275 for the best baseline, a state reward without report terms (+1.5 pp [0.6, 2.6]; the effect rests on a few tasks, and both PCSTR-d seeds give identical outcomes on all 400). Not registered, and between rewards of different form: the warm start scores 0.6425 and a tool-trusting reward 0.640 (PCSTR-d − tool-trusting +20.25 pp [16.4, 24.3]); on strict VSC, which also forbids unrequested writes, the tool-trusting reward falls to 0.424. With two rewards of identical form from one warm start (exploratory, development split), test-IID VSC is 0.825 against 0.605, and sampled refund read-back rises from 0.25 to 0.71 (state reward) and 0.72 (PCSTR-d) but falls to 0.10 under the acknowledgement reward. In a pre-registered intervention, removing refund timeouts and unrecoverable refunds restores read-back to 0.669, removing 60 random tasks leaves it at 0.000, and the state reward reads back 1.000 on both pools (interaction +0.678 [0.605, 0.750]; interval over tasks, one training seed per cell). An LLM judge (Aya Expanse 32B with the AuditKIT TaskCompletion prompt) given the database's receipt ledger still accepts 71% (213 of 298) of the false completions the ledger exposes. The behaviour is fragile: when a timeout can fail before the commit, PCSTR-d scores 0.176 and the state reward 0.118 VSC against 0.71 for the tool-trusting agent (stress test, 34 tasks, seed 0). Read-back was present in the supervised demonstrations, so these results show GRPO selecting or suppressing an existing behaviour; we do not claim that GRPO learns verification.

## Deliverables

| PS9 deliverable | Where |
|---|---|
| Interactive write-up (website) | https://atharvgaur18-actionproof.static.hf.space/index.html (Space: https://huggingface.co/spaces/atharvgaur18/actionproof) |
| Code (this repository) | https://github.com/atharvgaur1845/actionproof-v2 |
| HF procurement-study collection (30 LoRA adapters, ProcureSim splits, evaluation traces) | https://huggingface.co/collections/atharvgaur18/actionproof-procurement-study-6b0321cd11bad297aead6e81 |
| HF models collection (29 LoRA adapters) | https://huggingface.co/collections/atharvgaur18/actionproof-adapters-6af7fb12592ac424c2086c04 |
| HF datasets collection (benchmark, tool-call logs and traces) | https://huggingface.co/collections/atharvgaur18/actionproof-benchmark-traces-demo-6af7fb25037f3d7eaf294e22 |
| Write-up: metrics, methods, results, limitations | `docs/ARTICLE_DRAFT.md` (abstract `docs/ABSTRACT.md`) |
| Agentic trajectory metrics (task success, tool-call accuracy, average latency) | `results/agentic_metrics.md` (`scripts/agentic_metrics.py`); latency there is batched seconds per episode; single-request p50 / p95 per episode in `results/latency_single.md` (A100, batch 1) |
| Reproduction | Section "Reproduce" below |

- **Track:** 3, Agentic Fine-tuning.
- **Toolkit chain:** AgentTune → AuditKIT, plus CuratorKIT.
- **Models:** Cohere Tiny Aya Global (policy) and Aya Expanse 8B / 32B (teacher and judges).
- **Novelty and related work:** `docs/research/novelty_scouts_2026-11-11.md`.
- **Environment v2** (`PREREGISTRATION_A10_ADDENDUM.md`, `results/a10/v2_env.md`): scoped idempotency and timeouts before the commit. Existing results unchanged on the same tasks. In the stress test with timeouts before the commit, PCSTR-d scores 0.176 and the state reward 0.118 VSC against 0.71 for the tool-trusting agent (34 tasks, seed 0): they read, then report success without retrying.
- **Latest amendments:** `PREREGISTRATION_A10.md` (strict metric SVC: state arms SVC = VSC, PCSTR-d − tool_trust +41.9 pp on SVC; dependency-aware reward, offline: H-A10c holds, H-A10d fails, `results/a10/`), `PREREGISTRATION_A9.md` (grounded judge audit, `results/judge/grounded_audit.md`; task 421 not run), and an independently implemented verifier (pre-registered metric specification, no shared verification code) that reproduces our VSC on all 4,000 confirmatory episodes, 400 tasks × 10 runs (`results/reference_verifier.md`; strict VSC not re-implemented). Before them, `PREREGISTRATION_A8.md` and its addenda. X1 and the placebo are **completed** (`results/a8/x1_readback.md`); seed 1 of the A7 state arm (task 415) reproduces seed 0. Addendum 2 (`PREREGISTRATION_A8_ADDENDUM2.md`, registered 2026-11-14 before either run): task 417 (state reward on the X1 pool) and task 418 (A7 ack reward, seed 1); both finished: the state reward on the X1 pool reads back 152/152, interaction +0.678 [0.605, 0.750] (`results/a8/x1_interaction.md`); the ack arm's seed 1 reads back 0/162 like seed 0. All three registered predictions met. Before it, `PREREGISTRATION_A7.md` (results in `results/eval/a6/`, `results/eval/a7/`, corrected forced-prefix battery `results/a7/battery_v2.md`). Every deviation is logged in `docs/SPEC.md` §27.
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
for f in freeze_bench freeze_a4 freeze_a5 freeze_a6 freeze_a7 freeze_a8 freeze_a8b freeze_a8c freeze_a9 freeze_a10 freeze_a10b; do .venv/bin/python scripts/$f.py --check; done
```

The tests cover:
- fault semantics;
- the PCSTR ordering property (property-based);
- the reward sanity matrix;
- an oracle that solves 100% of every split;
- deterministic replay;
- the banking domain and the language variants.

Every check must print `OK`, except `freeze_a9` (D-A9-1, `judge_audit.py`) and `freeze_a5`, which reports `verifier_a5.py` and `a5_rewards.py`. That mismatch is the logged deviation D-A5-3 (a parsing fix after registration; SPEC §27, 2026-11-10). It was not re-registered on purpose. Checked from a fresh GitHub clone of `df25b8b` on 2026-11-15 19:10 IST (tests run with the project's existing `.venv`, not a fresh `setup_env.sh`; no CI is configured): 509 tests passed, 4 skipped; all eleven freeze checks OK except `freeze_a5` (D-A5-3) and `freeze_a9`, which reports `judge_audit.py` (D-A9-1, a one-line FP8 fix after registration; SPEC §27). Later commits change documents, result files and the rebuilt site only. Results on the website and in the paper are the files in this repository; the site footer names the commit it was built from.

### 3. Pipeline (each step is a queue task in `slurm/tasks/`)

| Step | Task | Command core |
|---|---|---|
| Teacher trajectories (Aya 32B, verify prompt) | `070` | `actionproof.eval.run_eval --model CohereLabs/aya-expanse-32b --prompt verify_a --split train --samples 8` |
| RFT v2 data + training | `087` | `actionproof.train.expert_demos` → `actionproof.train.rft_data --expert …` → `actionproof.train.rft` |
| Learnable-band curriculum | `091` | `run_eval --samples 8 --temperature 1.0` → `actionproof.train.curriculum` |
| GRPO arms (shared init, only the reward differs) | `150` to `157` | `slurm/grpo_arm.sh <arm> runs/rft_tiny_v2 data/curriculum/band_tiny_v2.jsonl v2` |
| Leaderboard | `300` to `302` | `run_eval` over 7 Cohere models × prompts × splits → `actionproof.eval.leaderboard` |
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
