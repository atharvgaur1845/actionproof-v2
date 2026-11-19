#!/bin/bash
# Two-lane GPU lease worker. qos_gpu_h200 allows one submitted job per user but two GPUs, so one job
# holds both GPUs and runs one copy of worker.sh per GPU (LANE=0,1; each lane sees only its own GPU).
#   sbatch -p gpu_h200_8 --gres=gpu:2 -c 8 --mem=280G --time=5-00:00:00 -J ap_w_h200x2 slurm/worker2.sh h200
#SBATCH --output=/scratch/pabitra/actionproof/logs/%x-%j.out
#SBATCH --signal=B:USR1@600
set -u
POOL=${1:?pool}
ROOT=/scratch/$USER/actionproof
IFS=, read -ra GPUS <<< "${CUDA_VISIBLE_DEVICES:-0,1}"
PIDS=()
for i in "${!GPUS[@]}"; do
  LANE=$i CUDA_VISIBLE_DEVICES=${GPUS[$i]} bash "$ROOT/slurm/worker.sh" "$POOL" \
    > "$ROOT/logs/${SLURM_JOB_NAME}-${SLURM_JOB_ID}-lane$i.out" 2>&1 &
  PIDS+=($!)
done
trap 'kill -USR1 "${PIDS[@]}" 2>/dev/null' USR1
for p in "${PIDS[@]}"; do
  while kill -0 "$p" 2>/dev/null; do wait "$p"; done
done
echo "worker2 exiting $(date +%T)"
