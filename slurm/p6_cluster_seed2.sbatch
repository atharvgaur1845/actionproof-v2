#!/bin/bash
#SBATCH --output=/scratch/pabitra/p6/logs/%x-%j.out
# P6 addendum 1: seed 2 of both arms on one sharanga H200, in sequence on the same GPU, in /scratch/pabitra/p6.
# Runs the registered task scripts unchanged (train, then eval, per arm). claim.sh is skipped (AP_HF=1) because
# it deletes a task file when it sees a duplicate claim; a directory lock prevents a second copy from running.
#   sbatch -p gpu_h200_8 --gres=gpu:1 -c 4 --mem=140G --time=06:00:00 -J p6_seed2 slurm/p6_cluster_seed2.sbatch
set -uo pipefail
cd /scratch/pabitra/p6
mkdir -p logs queue
mkdir queue/seed2_lock 2>/dev/null || { echo "another seed-2 job holds the lock"; exit 0; }
echo "slurm:${SLURM_JOB_ID:-?}:$(hostname):$(date -Is)" > queue/seed2_lock/owner
export AP_HF=1 AP_ROOT=/scratch/pabitra/p6 PYTHONPATH=/scratch/pabitra/p6
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
.venv/bin/python -c "import actionproof, sys; print('actionproof from', actionproof.__file__)"
rc=0
for job in can_s2 surf_s2; do
  for t in 2300_p6_$job 2310_p6_eval_$job; do
    echo "=== $t $(date -Is)"
    bash slurm/tasks/$t.sh; r=$?
    echo "=== $t rc=$r $(date -Is)"
    [ "$r" = 0 ] || { rc=$r; break 2; }
  done
done
echo "$rc" > queue/seed2_lock/rc
exit $rc
