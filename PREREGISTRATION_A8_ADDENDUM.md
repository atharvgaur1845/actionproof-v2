# Pre-registration, Amendment 8 addendum: placebo control for X1

Registered 2026-11-13, before X1 (task `414p`) or the placebo (task `414q`) had started. At registration no X1 output existed: no `runs/grpo_tiny_a8x1_*` directory and no X1 evaluation summary. The commit time of this file is the registration time. The design comes from the Fable 5.2 triage of an external audit (SPEC §27, 2026-11-13).

`PREREGISTRATION_A8.md` and every file in `results/a8_hashes.json` are unchanged. The new files are hashed in `results/a8b_hashes.json` by `scripts/freeze_a8b.py`.

## 0. Why

X1 removes the 60 refund-timeout and refund-UR tasks from pool B. That changes two things at once:
- which task classes the acknowledgement reward trains on;
- the pool size, which goes from 400 to 340 tasks, or 2.0 to 2.35 epochs over 200 steps.

X1 also differs from its control (task 412) in being time-sliced and logged. So a positive X1 result could come from any 340-task pool, not from those two classes. The placebo separates the two explanations.

## 1. Placebo X1

| | X1 (`414p`) | Placebo (`414q`, new) |
|---|---|---|
| Reward, warm start, GRPO config, seed, turns, evaluation | A7 `ack_outcome`, `rft_tiny_a7`, as task 412, seed 0 | same |
| Tasks | pool B minus the 60 target tasks (340) | pool B minus 60 **random non-target** tasks (340); all 60 target tasks kept |
| Script | `slurm/a8_x1.sh` | `slurm/a8_x1p.sh` (copy; only the task list, tag and freeze check differ) |
| Rollout log | on | on |
| Run tag | `a8x1_ack_outcome_s0` | `a8x1p_ack_outcome_s0` |

**Selection rule** (deterministic). The removed set is `random.Random("actionproof-a8/placebo/0").sample(sorted(non_target), 60)`, where `non_target` is the 340 task ids of `data/bench/a8/x1_task_ids.jsonl`. The kept tasks are written in pool-B file order to `data/bench/a8/x1p_task_ids.jsonl` (sha256 `6251343992ed938d34477eea1e5d9860e89fbe888d1fd6d30bd81a644361a269`).
- **Removed by class:** OK 12, close-timeout 10, F5 7, refund-phantom 7, audit-timeout 6, audit-phantom 6, UR-close 5, close-phantom 4, UR-audit 3.
- **Target share:** the target tasks rise from 15% (60/400) to 17.6% (60/340) of the pool.

**Order.** X1 → placebo → 415 (queue name order: `414p`, `414q`, `415`). The placebo task file skips itself, printing `SKIP` and exiting 0, if X1's `test_iid` summary exists with PWV_refund ≤ 0.10 when it would start. In that case it is reported as "not run (X1 rejected)". `# after: 414p` is not used: a time-sliced X1 reaches `done/` after its first slice.

## 2. Predictions and decision rules

The primary metric is the same as X1: greedy `test_iid` PWV_refund from `results/eval/a6/a8x1p_ack_outcome_s0__neutral__test_iid.summary.json`.

| ID | Prediction | Rule |
|---|---|---|
| **H-A8d** | Placebo PWV_refund ≤ 0.10, like the control (412: 0/169) | ≤ 0.10: the control holds. ≥ 0.30: the control fails. In between: inconclusive. |
| Joint reading 1 | X1 ≥ 0.30, placebo ≤ 0.10, and the paired task bootstrap (10k) 95% CI of PWV(X1) − PWV(placebo) excludes 0 | "Class-specific removal supported (one seed per cell)". |
| Joint reading 2 | X1 ≥ 0.30 and placebo ≥ 0.30 | H-A8 is met by its own rule but is not attributable to the two classes. The mechanism wording becomes "consistent with the step-0 audit; not established". |
| Interaction on PWV | I = [PWV(ack, filtered) − PWV(ack, full)] − [PWV(state, filtered) − PWV(state, full)]. The filtered-pool state arm is not run. With the seed-0 full-pool cells at 0/169 (412) and 152/152 (411), I = PWV(X1) + (1 − PWV(state, filtered)) ≥ PWV(X1) for any value of the missing cell | Reported as a lower bound, with the missing cell named. No between-seed inference. |
| **H-A8e** (descriptive) | In logged training rollouts, steps ≤ 50, per task class: g = mean reward(readers) − mean reward(non-readers), rescored with the arm's own reward (`scripts/a8_rollout_class.py`, mixed (step, task) groups). Placebo: g ≤ −0.3 on refund timeout and on refund UR, and \|g\| < 0.3 elsewhere. 415 (state arm `pcstr_d_noreport`, seed 1, logged): \|g\| < 0.3 on refund timeout | Fewer than 20 episodes on either side gives "insufficient data". The channel × class contrast is g_ack − g_state on refund timeout, with a task-cluster bootstrap (10k). Refund UR is not a clean channel contrast: the state reward `pcstr_d_noreport` also pays readers less there at step 0 (0.04 vs 0.17, A7 gate). |

**Freeze rule.** An arm with no `test_iid` summary at 2026-11-15 18:00 IST is reported as "registered, not completed". There are no partial reads and no checkpoint evaluations.

**Fallback.** If no H200 worker has started by 2026-11-14 08:00 IST and an A100 worker is running, `414p` may be moved to `queue/a100/pending`, logged as a deviation. Its self-re-queue path is h200, so re-queued slices are moved by hand. The hashed scripts are not edited.

## 3. Logging for 415 (deviation, Amendment 7 code untouched)

Task 415 had not started at registration. Both pending copies (h200 and a100) gain one line, `export AP_ROLLOUT_LOG=runs/grpo_tiny_a7_pcstr_d_noreport_s1/train_rollouts.jsonl`. It changes nothing in training: the logger is additive and off unless set. `slurm/a7_arm.sh` and the Amendment 7 hashes are unchanged.

## 4. What does not change

- X1's design, hypotheses and thresholds (`PREREGISTRATION_A8.md`).
- 415's Amendment 7 rules.
- No filtered-pool state arm, and no additional seeds: there is no GPU budget for them before the freeze.
