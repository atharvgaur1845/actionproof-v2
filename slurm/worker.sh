#!/bin/bash
# GPU lease worker. sharanga's queue is age-ordered, so we keep worker jobs queued and
# aging; once one starts it holds its GPU for the full wall time and runs task scripts
# dropped into   $Q/<pool>/pending/*.sh   (pool = h200 | h100 | a100).
#
# Task protocol: first line may declare `# minutes: N`; the worker only starts a task that
# fits in the remaining wall time. Tasks are claimed atomically by `mv` into running/, so
# several workers can share one pool. Logs go to $Q/<pool>/logs/<task>.log.
#
# Submit (examples):
#   sbatch -p gpu_h200_8 --gres=gpu:1 -c 6  --mem=140G --time=24:00:00   -J ap_w_h200 slurm/worker.sh h200
#   sbatch -p gpu_h100_4 --gres=gpu:1 -c 12 --mem=200G --time=2-00:00:00 -J ap_w_h100 slurm/worker.sh h100
#SBATCH --output=/scratch/pabitra/actionproof/logs/%x-%j.out
#SBATCH --signal=B:USR1@600
set -u
POOL=${1:?pool}
ROOT=/scratch/$USER/actionproof
Q=$ROOT/queue/$POOL
mkdir -p "$Q"/{pending,running,done,failed,logs}
export HF_HOME=/scratch/$USER/.cache/huggingface HF_HUB_OFFLINE=1
END=$(( $(date +%s) + $(squeue -h -j "$SLURM_JOB_ID" -o %L | awk -F'[-:]' '{ if (NF==4) print ($1*86400+$2*3600+$3*60+$4); else if (NF==3) print ($1*3600+$2*60+$3); else print ($1*60+$2) }') ))
STOP=0
trap 'STOP=1; [ -n "${CHILD:-}" ] && kill -USR1 "$CHILD" 2>/dev/null' USR1

echo "worker $SLURM_JOB_ID pool=$POOL node=$(hostname) gpu=$CUDA_VISIBLE_DEVICES until $(date -d @$END)"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader

while [ "$STOP" = 0 ]; do
  left_min=$(( (END - $(date +%s)) / 60 - 10 ))
  [ "$left_min" -le 5 ] && break
  task=""
  for f in $(ls -1 "$Q/pending/"*.sh 2>/dev/null | sort); do
    need=$(sed -n '1,5s/^# minutes: *\([0-9]\+\).*/\1/p' "$f" | head -1)
    if [ "${need:-60}" -le "$left_min" ] && mv "$f" "$Q/running/" 2>/dev/null; then
      task="$Q/running/$(basename "$f")"; break
    fi
  done
  if [ -z "$task" ]; then sleep 60; continue; fi
  name=$(basename "$task" .sh)
  echo "$(date +%T) start $name (left ${left_min}m)"
  ( cd "$ROOT" && exec bash "$task" ) > "$Q/logs/$name.log" 2>&1 &
  CHILD=$!; wait "$CHILD"; rc=$?; CHILD=""
  if [ "$rc" = 0 ]; then mv "$task" "$Q/done/"; else mv "$task" "$Q/failed/"; fi
  echo "$(date +%T) end   $name rc=$rc"
done
echo "worker exiting $(date +%T)"
