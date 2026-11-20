# K1: task-composition control for the acknowledgement reward

Registered 2026-11-20 18:00 IST, before any K1 run starts. The commit time of this file and of `results/k1_hashes.json` is the registration time. Every earlier registration is unchanged. This control was named in Addendum 2 (§ reward-signal analysis) as the follow-up if the reward-signal analysis left a competing explanation open. It did: in A12, 60–92% of R_A's informative training groups came from the lost-acknowledgement and never-commits classes, where reading the payment record lowers R_A (`results/s2/reward_signal_summary.json`).

## 1. Question

Is R_A's flat phantom read-back (0.00 in all five A12 seeds, identical to the warm start) caused by the acknowledgement channel itself, or by the training pool's composition: many tasks where reading is penalised under R_A?

## 2. Conditions

All runs use R_A, the A12 warm start (`runs/rft_proc/final`) and the A12 trainer and settings unchanged (`actionproof/train/grpo_proc.py`: 200 steps, 4 tasks × 8 rollouts, lr 2e-5, KL coefficient 0.02, temperature 1, LoRA rank 16). Only `--task-jsonl` differs.

| Condition | Pool | Size | Composition |
|---|---|---|---|
| Targeted (k1t) | `proc_train_k1t` | 223 | `proc_train_b` minus every lost_ack and never_commit task |
| Placebo (k1p) | `proc_train_k1p` | 223 | `proc_train_b` minus 177 tasks chosen uniformly at random (seed 113) |

Both pools come from `scripts/make_k1_pools.py` and are hashed. Seeds 0, 1 and 2 per condition (six runs). The existing A12 R_A seeds 0–4 are the reference, trained on the full pool. They are not re-run.

## 3. Endpoints

- **Primary:** greedy phantom read-back on `proc_conf` (the A12 H2 metric), per seed, for each K1 model. Each model is evaluated once on `proc_conf`, `proc_combo` and `proc_stress` through the confirmatory machinery, with its own access-log entry.
- **Key secondary:** sampled read rate after the pay write over training, and per-class informative-group shares, from the rollout logs (`scripts/a12_reward_signal.py`, applied unchanged).
- **Secondary:** VSC and false success on `proc_conf`; development read-back at steps 100 and 200 on `proc_dev`; the A17 cross-domain evaluation (registered separately).

## 4. Prediction

Under R_A, reading earns no reward in any class left in the targeted pool. On phantom tasks, both proceeding and reading-then-retrying end with an ok acknowledgement. Prediction:
- targeted removal stops the *decline* in sampled read rate seen in A12 (about 0.12 to 0.05);
- it does **not** raise greedy phantom read-back above the warm start (0.00 on `proc_conf`).

## 5. Interpretation (fixed now)

- **Targeted raises read-back, placebo does not:** for at least 2 of 3 seeds, k1t phantom read-back exceeds 0.10 with a task-bootstrap 95% interval excluding 0, and k1p stays within 0.10 of 0. Then the task mix explains part of A12, and the paper narrows its claim to "the acknowledgement reward, on this pool".
- **Both stay at 0:** composition does not explain R_A's flat read-back. The channel explanation stands for this training setup.
- **Both rise:** pool size or exposure, not composition, matters.
- **Anything else:** inconclusive, reported as such.

With 3 seeds, the exact sign-flip test has 1/8 one-sided resolution. No seed-level significance is claimed. Results are reported per seed.

## 6. Protocol

- The checkpoint evaluated is step 200. A run that does not reach step 200 is reported as missing.
- Runs are distributed between the Slurm H200 worker and HF Jobs H200 (same code and the same pinned requirements, `requirements.lock`). The executor is logged per run.
- Deviations go to `docs/stage2/DECISIONS.md` with their time.
