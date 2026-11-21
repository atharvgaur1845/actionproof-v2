# S3: recovery-start GRPO (RS-GRPO) against state-reward GRPO (A12 R_V)

Drafted 2026-11-21. The registration time is the commit time of this file and `results/s3_hashes.json`, which is before any model is evaluated on the confirmatory splits below. Design and development history: `docs/stage3/RESEARCH_REPORT.md`, `docs/stage3/LITERATURE.md`. Every earlier registration and result is unchanged.

## 1. Development history (disclosed; development data only)

- **Pilot** (100 steps, seed 0), on `s3_rec_dev` (150 recovery starts) and `proc_dev` (first 150 tasks). Four candidates were tested: A (safety-first lexicographic reward), F (recovery starts with the R_V reward), F+A, and F+D (Lagrangian). Only F improved the recovery-start score (`results/s3/dev`). A, F+A and F+D did not beat F, and A did not beat A12 R_V s0.
- **Stage B** (200 steps, seeds 0–1): F with 50% recovery starts, against 25%.
  - F 50%, both seeds: `s3_rec_dev` VSC 0.81 (A12 R_V s0: 0.60); violations 0.03 (0.15); `proc_dev` VSC 0.86 (0.847).
  - F 25%: weaker.
- **Selection:** the pre-set rule in the report (§6) selected F 50%.
- **Pre-registration training:** seeds 0 and 1 of the selected configuration were trained before this registration. Neither has been evaluated on any confirmatory split. Seeds 2–4 are trained after it.

## 2. Algorithm (frozen)

**RS-GRPO** (`actionproof/stage3`; entry point `python -m actionproof.stage3.train --reward rv`) uses the A12 GRPO trainer and settings unchanged:
- warm start `runs/rft_proc/final`;
- 200 steps, 4 prompts × 8 rollouts per step;
- lr 2e-5, KL 0.02, temperature 1, LoRA rank 16, up to 16 turns.

The reward is A12's R_V. **The only change is the start-state distribution.** The pool is 200 `proc_train_b` tasks (`s3_trainb_half`) plus 200 recovery starts (`s3_rec_train_half`). A recovery start replays a scripted prefix that ends:
- at the acknowledgement of a faulted write (phantom; partial at ratio ¼, ½ or ¾; lost acknowledgement; duplicate delivery; complete; never commits);
- or after its authoritative read;
- or after a fully completed workflow.

The replayed turns are prompt, so no loss falls on them. Run tags are `s3_rv_mix_s<seed>`, seeds 0–4.

**Baseline:** A12 R_V seeds 0–4 (frozen step-200 adapters; `results/a12/frozen_models*.json`): the same warm start, trainer, settings, reward and number of rollouts, trained on 400 `proc_train_b` tasks. Run tags are `a12_rv_s<seed>`. **References:** the warm start; A12 R_A seeds 0–2; the scripted oracle and never-read policies (`results/s3/scripted_conf_calibration.json`).

## 3. Confirmatory splits (generated at registration; hashed)

| Split | n | Seed | Content |
|---|---|---|---|
| `s3_conf_rec` | 300 | 212 | recovery starts (same start mix as training, fresh tasks) |
| `s3_conf_full` | 300 | 211 | full episodes: A12 classes, partial at ratio ¼/½/¾, plus duplicate delivery on pay |
| `s3_conf_stress` | 100 | 213 | duplicate delivery on pay, or lost acknowledgement then a lagging read |
| `s3_it_ops`, `s3_care_admin` | 120 each | 231, 232 | fresh instances of the OpsSim templates (transfer) |

All PO ids and task ids are disjoint from every earlier split (asserted). Evaluation uses `scripts/s3_eval.py` (ProcureSim splits) and `scripts/eval_opsim.py --split` (OpsSim): greedy, 16 turns, 256 tokens, one run per model per split.

## 4. Hypotheses and tests

The unit is the task. Per seed, RS-GRPO seed s is compared with A12 R_V seed s on the same tasks (paired). Statistics: `scripts/s3_stats.py`, with paired task bootstrap (10,000 resamples), exact McNemar, exact sign-flip across five seeds, and a seed-averaged bootstrap that shares one task draw across all seeds.

- **H1 (primary, superiority):** VSC on `s3_conf_rec` is higher for RS-GRPO. **Supported** if the per-seed difference is positive in at least 4 of 5 seeds **and** the seed-averaged 95% interval excludes 0.
- **H2 (primary, non-inferiority):** VSC on `s3_conf_full` is not worse. **Supported** if the seed-averaged lower 95% bound of (RS − R_V) is above −0.05.
- **The algorithm is said to improve on R_V only if H1 and H2 both hold.** No multiplicity correction applies to this conjunction; each test is at 0.05.
- **Secondary (reported per seed and seed-averaged; no decision rule):**
  - violation rate (duplicate, collateral, wrong amount or payee) and false completion on `s3_conf_rec`, `s3_conf_full` and `s3_conf_stress`;
  - VSC on `s3_conf_stress`;
  - per start type on `s3_conf_rec` (in particular "complete" and "done": not acting again);
  - honest failure and escalation rate (collapse check);
  - policy calls and reads per episode (efficiency);
  - OpsSim transfer: phantom read-back, VSC and false success on `s3_it_ops` and `s3_care_admin`.
- **Not claimed:** any mechanism, any safety guarantee, and any effect of the safety-first or Lagrangian rewards, which were not carried forward. Their development results are reported as exploratory.

## 5. Protocol

- Each model is evaluated once per split.
- Runs use the Slurm H200 worker and HF Jobs; the executor is logged.
- Missing runs are reported as missing; no seed is dropped.
- Deviations go to `docs/stage2/DECISIONS.md` with their time.
