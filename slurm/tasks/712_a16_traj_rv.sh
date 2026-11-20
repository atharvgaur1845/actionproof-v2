# minutes: 240
# A16 trajectory (PREREGISTRATION_A15 §4.1-4.2): margins and probes, warm start and rv seeds 0-2 at steps 50-200.
set -uo pipefail
source slurm/claim.sh || exit 0
source slurm/env.sh
export MASTER_PORT=29712
.venv/bin/python scripts/freeze_a15.py --check || exit 1
mkdir -p logs results/a15 results/a16 results/eval/a15
[ -f runs/rft_proc/merged/config.json ] || .venv/bin/python -m actionproof.train.merge runs/rft_proc/final CohereLabs/tiny-aya-global > logs/a15_merge.log 2>&1 || { tail -20 logs/a15_merge.log; exit 1; }
.venv/bin/python scripts/a16_mech.py trajectory --n-test 100 --models warm=runs/rft_proc/merged,rv_s0_50=runs/a12_proc_rv_s0/checkpoint-50,rv_s0_100=runs/a12_proc_rv_s0/checkpoint-100,rv_s0_150=runs/a12_proc_rv_s0/checkpoint-150,rv_s0_200=runs/a12_proc_rv_s0/checkpoint-200,rv_s1_50=runs/a12_proc_rv_s1/checkpoint-50,rv_s1_100=runs/a12_proc_rv_s1/checkpoint-100,rv_s1_150=runs/a12_proc_rv_s1/checkpoint-150,rv_s1_200=runs/a12_proc_rv_s1/checkpoint-200,rv_s2_50=runs/a12_proc_rv_s2/checkpoint-50,rv_s2_100=runs/a12_proc_rv_s2/checkpoint-100,rv_s2_150=runs/a12_proc_rv_s2/checkpoint-150,rv_s2_200=runs/a12_proc_rv_s2/checkpoint-200 > logs/a16_traj_rv.log 2>&1 || { tail -30 logs/a16_traj_rv.log; exit 1; }
