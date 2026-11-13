# Pre-registration, Amendment 8: does an acknowledgement reward remove read-back only where reading costs acknowledgements?

Registered 2026-11-13, before any Amendment 8 GPU run. The design comes from the Fable 5.2 research-director review of 2026-11-13 (SPEC §27). Deadline: 2026-11-15 midnight IST; experiment freeze 2026-11-15 18:00 IST.

Amendments 0–7 and their hashes are unchanged. The new files are hashed in `results/a8_hashes.json` by `scripts/freeze_a8.py`. One shared module gains an additive change: `actionproof/rollout/batched_vllm.py` writes training episodes to disk when `AP_ROLLOUT_LOG` is set. It is off by default, and no earlier run set it.

## 0. Why

The article (§3) says that under an acknowledgement-grounded goal term, a read earns no expected advantage, so verification drifts away, possibly through the turn budget.

The warm starts' own step-0 samples support a stronger mechanism (`results/a8/ack_class_audit.md`, recomputed from the gate files):
- Under `ack_outcome`, episodes that read back the refund are paid *less* than non-readers, and only in two task classes.
- **Refund timeout** (the refund committed; the acknowledgement says error). A reader sees the refund landed and correctly does not re-issue it, so the last acknowledgement stays "error". Reward, readers vs non-readers: A6 0.08 vs 0.62; A7 0.00 vs 0.36.
- **Refund unrecoverable** (the refund never commits). A reader sees it will not land and stops, so later writes are never acknowledged. Reward: 0.25 vs 1.00 in both warm starts.
- **All other classes:** the step-0 selection on reading is ≥ 0. Summed over groups: A6 +3.85, A7 +8.85.

**Thesis under test.** Given an evidence response fixed by SFT, an acknowledgement reward gives reading back a negative value only on the task classes where acting on the evidence forgoes success acknowledgements, and GRPO removes the read because of those classes.

**Competing reading.** Removal comes from task-independent costs of reading (turns, length) or from drift. It would happen whatever the task mix.

## 1. Experiment X1

Task 412 (`grpo_tiny_a7_ack_outcome_s0`) is the control. X1 is the same run with one change: the training task list.

| | Control (412, existing) | X1 (new) |
|---|---|---|
| Warm start | `runs/rft_tiny_a7` | same |
| Reward | `ack_outcome` (`rewards/ack_outcome.py`, unchanged) | same |
| Tasks | `a4_train_B`, 400 | `a4_train_B` minus the 44 RP `issue_refund` F4_timeout tasks and the 16 UR `issue_refund` tasks = **340** (`data/bench/a8/x1_task_ids.jsonl`, via the existing `--task-file` flag) |
| GRPO | 200 steps × 4 prompts × 8 rollouts, lr 2e-5, β 0.02, 18 turns, seed 0, batched vLLM | same |
| Evaluation | `slurm/a6_eval.sh` (all A6 splits) | same |
| Training rollouts | not logged | logged (`AP_ROLLOUT_LOG`) |

Run script: `slurm/a8_x1.sh`. It is time-sliced: it trains until 15 minutes before the worker's wall time, re-queues itself, and resumes from the last checkpoint. Resuming from a checkpoint is the existing `grpo.py` behaviour.

**Known differences from the control.**
- The pool is smaller (340 vs 400 tasks), so 200 steps cover 2.35 epochs instead of 2.0.
- If a time slice ends mid-run, the steps after the last checkpoint are re-sampled on resume.
- The rollout log keeps both copies of those re-sampled steps; the analysis keeps the last block per step.

## 2. Hypotheses and decision rules

Primary metric: greedy post-write refund verification on `test_iid`. This is `a4["pwv_issue_refund"]["rate"]` in `results/eval/a6/a8x1_ack_outcome_s0__neutral__test_iid.summary.json`. `test_iid` runs in the training schema, the plain `EcomEnv`, so D-EVAL-1 does not apply.

Reference values:

| Run | PWV (refund) | n |
|---|---|---|
| A7 init | 0.60 | 155 |
| Control (412) | 0.00 | 169 |
| A7 state arm `pcstr_d_noreport` s0 | 1.00 | 152 |

| ID | Hypothesis | Decision rule |
|---|---|---|
| **H-A8 (primary)** | With the two classes removed, the acknowledgement reward does not remove read-back | X1 PWV ≥ 0.30: **supported**. 0.10 < PWV < 0.30: **inconclusive**, reported as such. PWV ≤ 0.10: **rejected**; the negative-value mechanism is withdrawn as the explanation and the article keeps its current wording |
| H-A8a (secondary) | The step-0 audit's sign holds: on the X1 pool the selection on reading is positive (+8.85), so read-back rises above the warm start | X1 PWV > 0.60, with a Wilson 95% CI excluding 0.60. Exploratory |
| H-A8b (secondary, exploratory) | Audit sign predictions for task 415 (`pcstr_d_noreport` s1 on the A7 warm start, full pool): reading (+17.39) and the demonstrator key (+16.06) both rise from the init (both 105/176 on `test_iid` greedy) | Signs correct for both traits. The two markers coincide in the A7 init (105/176 each), so this is one prediction, not two. It is not a test of the audit as a method |
| H-A8c (secondary) | The mechanism is visible in training | In X1's logged rollouts, readers and non-readers earn similar `ack_outcome` reward in every kept class (no class with a reader deficit ≥ 0.3 over ≥ 20 episodes of each). Descriptive |

Secondary metrics: X1 VSC on `test_iid`; the demonstrator-key rate; HFR on `e_unrecoverable` (evaluation schema, D-EVAL-1 applies); zero-variance group share.

**Statistics.**
- Wilson 95% intervals per rate.
- Task-level paired bootstrap (10k resamples) for X1 vs control on `test_iid` PWV, over tasks with a success-acknowledged refund in both runs.
- One seed: no between-seed inference. The comparison is between two training runs that differ in one factor.

**Caveats fixed now.**
- The step-0 audit is weak where it matters most. On the full pool the A7 audit sum is only −0.74 (−0.012 per mixed group), yet the control went from 0.60 to 0.00, so training dynamics amplify a small initial signal.
- The audit uses evaluation-schema gate samples, groups of 4 and 100 tasks.
- A positive X1 result shows that removing those classes is sufficient to keep read-back. It does not show that their presence is necessary for every acknowledgement reward: other definitions of an ack reward are out of scope.

## 3. What does not change

- No other arm, split, threshold or registered hypothesis of Amendments 0–7.
- Task 415 stays queued behind X1 with its own Amendment 7 rules.
- The verdicts on A7 are not reopened by this amendment.
