# P6 addendum 1: where the six runs execute

Written 2026-11-29, after registration (`results/p6_hashes.json`, registered 2026-11-28T19:51:42Z, commit `8e8abd7`)
and before any P6 training or evaluation has started. It changes only where and on what hardware the registered tasks
run. Arms, seeds, data, reward, steps, optimizer settings, evaluation, hypotheses, decision rules, the false-completion
guardrail and the reporting cutoff are unchanged. No registered file is edited, and `scripts/freeze_p6.py --check`
still runs before every training and evaluation task.

## Reason

The HF Jobs budget for P6 is at most $50. The same training recipe (P1 arm D: R_V, the same two pools, 200 steps from a
warm start) took 1 h 39 min per task on the cluster H200 (`1154_p1_d_s0`) and about 3 h 35 min on the cluster A100
(`1155_p1_d_s1`). Six HF H200 jobs of about 2 h 15 min each (setup, training, merge and seven-split evaluation) would
cost about $68, which is over the budget.

## Change

| Runs | Where | Launch |
|---|---|---|
| `P6-CAN` and `P6-SURF`, seeds 0 and 1 (4 jobs) | HF Jobs, 1× H200 each, 2 h 30 min timeout (at most 4 × 2.5 h × $5 = $50) | `scripts/p6_hf_launch_a1.sh`, a copy of the frozen `scripts/p6_hf_launch.sh` that differs only in timeout and job list |
| `P6-CAN` and `P6-SURF`, seed 2 | sharanga cluster, one H200, both arms in sequence on the same GPU, in `/scratch/pabitra/p6` | `slurm/p6_cluster_seed2.sbatch` |

Each job runs the registered task scripts unchanged: `2300_p6_<arm>_s<k>` (train and merge), then
`2310_p6_eval_<arm>_s<k>`. The HF jobs install the environment from the pinned `requirements.lock` (as in
`slurm/hfjob_run.sh`); the cluster job uses the cluster environment built from the same lock file, with
`PYTHONPATH` set so the registered code in `/scratch/pabitra/p6` is the code that runs. Seeds are paired within a
platform: both arms of a seed run on the same platform and GPU type. The GPU model and the start and end time of every
job are recorded in its log.

## Consequences

- If an HF job hits its timeout or fails, the seed is not re-run on HF beyond the $50 cap. It may be re-run once on the
  cluster with the same task scripts, and the failure is reported.
- The seed-2 cluster job waits for a free H200. If it, or any HF job, has not finished by 2026-11-29 17:29 IST, the
  registered cutoff rule applies: P6 is not reported in the submitted papers, and no partial or per-seed result is
  quoted. The full registered analysis is reported later, whatever the outcome.
