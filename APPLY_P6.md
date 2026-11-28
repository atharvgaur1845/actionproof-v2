# Install and execute P6

This patch adds a preregistered follow-up study to the ActionProof repository. It does not alter `submission-candidate`
files or any existing model/data/results. Apply at the repository root:

```bash
git switch -c research/p6-surface-randomized-grpo
git apply /path/to/actionproof_p6.patch
```

The ZIP is also provided for manual copying if preferred.

## Before training

The patch deliberately refuses to run training until the experiment is frozen. On the cluster, from the repository
root:

```bash
pytest tests/test_p6_stats.py -q
.venv/bin/python scripts/freeze_p4.py --check
.venv/bin/python scripts/freeze_p6.py --register
git add results/p6_hashes.json
git commit -m "Register P6 surface-randomized RL continuation"
```

`--register` verifies and hashes the three `Dw_aug` initial adapters, all P4 baseline outputs, P6 training pools,
evaluation splits, the trainer/randomizer/evaluator code, and this preregistration. Do not run it until the three P4
adapters and all P4 `Dwaug_s*` evaluation files are present. If anything is missing, stop and resolve the artifact
path; do not weaken the manifest.

## Launch

Use the repository's existing Slurm task queue. The six training task files are:

- `slurm/tasks/2300_p6_can_s0.sh` through `...s2.sh`
- `slurm/tasks/2300_p6_surf_s0.sh` through `...s2.sh`

The six evaluation tasks are `slurm/tasks/2310_p6_eval_{can,surf}_s{0,1,2}.sh`; each evaluates one arm on all
seven registered splits. Run `slurm/tasks/2320_p6_stats.sh` only after all six model/seed combinations have complete
outputs. Alternatively, launch the task files using the same queue mechanism used for P1/P4/SR1.

The commands can also be run directly (with the repository's `.venv` active). Training examples:

```bash
.venv/bin/python -m actionproof.stage3.train --reward rv \
  --pools data/bench/proc/s3_trainb_half.jsonl,data/bench/proc/s3_rec_train_half.jsonl \
  --seed 0 --steps 200 --init-adapter runs/p4_dwaug_s0/final \
  --out runs/p6_canonical_s0 --use-vllm --vllm-mem 0.30

.venv/bin/python -m actionproof.stage4.train_variants --surface-random --reward rv \
  --pools data/bench/proc/s3_trainb_half.jsonl,data/bench/proc/s3_rec_train_half.jsonl \
  --seed 0 --steps 200 --init-adapter runs/p4_dwaug_s0/final \
  --out runs/p6_surface_s0 --use-vllm --vllm-mem 0.30
```

These are matched continuations from the same P4 adapter. Do not change steps, pools, beta, LR, reward, surface
randomization fraction, number of generations, or evaluation samples after registration.

## Output and decision

- Training adapters/merged models: `runs/p6_{canonical,surface}_s{0,1,2}/`
- Evaluation traces: `results/p6/P6{C,S}_s{0,1,2}__<split>.jsonl`
- Registered analysis: `results/p6/p6_summary.json`

A successful P6 result requires API superiority over canonical GRPO (H1) **and** retention of P4's API, workflow and
canonical performance (H2–H4), with the false-completion guardrail reported. A failed H1 or any failed retention
hypothesis is not to be hidden or rewritten as a success. No first-place or 10/10 claim is implied until results are
run and audited.
