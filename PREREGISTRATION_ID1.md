# ID1: does the state reward select read-back from the warm start's repertoire, or create it?

Registered 2026-11-21, before any model named here is trained. The registration time is the commit time of this file and `results/id1_hashes.json`. No earlier registration or result changes.

## 1. Question

In A12, the A12 warm start (`runs/rft_proc`) was trained on demonstrations of which 30% verify (the oracle reads after every write) and 70% trust the acknowledgement. Under greedy decoding it never reads after a phantom write (phantom read-back 0.00 on `proc_conf`), but it samples reads at temperature 1. R_V training made read-back the default in 5 of 5 seeds (1.00 on `proc_conf`).

Two readings fit that result:
- R_V **selected** a behaviour the warm start already sampled;
- or R_V would **create** read-back from a policy that never demonstrates it.

This is the identification gap raised by both academic reviews (`docs/stage3/ACADEMIC_REVIEW.md` W1; `ACADEMIC_REVIEW_POST.md` §4 item 4).

## 2. Design

- **Warm start `rft_proc_v0`.** It is identical to `rft_proc` except that the verifying share of demonstrations is 0 (`proc_demos --verify-share 0`). So:
  - the same `proc_demo` tasks, demonstration seed (1201), SFT script, settings (2 epochs) and base model;
  - every demonstration trusts the acknowledgement.
  - The tool list in the system prompt still offers every read tool.
- **R_V runs.** `grpo_proc --arm proc_rv` from `rft_proc_v0`, seeds 0 and 1, on `proc_train_b`. All A12 settings are unchanged (200 steps, 4 prompts × 8 generations, lr 2e-5, β 0.02, T = 1).
- **Compute.** The Slurm A100 lane (tasks 980–982), with copies on the H200 lane as a fallback. The executor of each task is recorded in its claim file. This hardware differs from A12, which is one reason the comparison with A12 below is a reference, not a test.

## 3. Measurements

| Quantity | Model | Source |
|---|---|---|
| Sampled read behaviour of the warm start (100 `proc_train` tasks × 4 samples, T = 1) | `rft_proc_v0` | the A12 gate procedure |
| **Primary:** greedy phantom read-back on `proc_conf` | `id1_rv_s0`, `id1_rv_s1` | `eval_proc.py`, once per model |
| VSC, false success on `proc_conf` | the warm start and both R_V runs | same |
| Read rate in training rollouts by step | both R_V runs | rollout logs |
| Phantom read-back on `proc_dev` at steps 100 and 200 | both R_V runs | development |

## 4. Interpretation, fixed now

| Primary result | Reading |
|---|---|
| Both seeds ≤ 0.10 | **Selection.** R_V does not create read-back that the warm start never demonstrates. A12's effect depends on the warm-start repertoire. |
| Both seeds ≥ 0.50 | **Discovery.** R_V creates read-back from exploration alone. A12's effect does not need verifying demonstrations. |
| Otherwise | Mixed: seed-dependent discovery; reported per seed. |

- **Reference, not a test:** A12 R_V seeds 0–4 on `proc_conf` (phantom read-back 1.00 in each; `results/eval/a12/conf/`) and the A12 warm start's sampled gate.
- Two seeds is a small sample. The reading is stated per seed and called preliminary.

## 5. Not claimed

- This is one base model, one demonstration set and one reward. It does not test other ways of creating a repertoire, such as prompting or a different base model.
