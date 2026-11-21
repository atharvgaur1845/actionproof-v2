# minutes: 200
# Stage 3 development stage B (200 steps; dev splits only): F, 25%% recovery starts, seed 1
set -uo pipefail
source slurm/claim.sh || exit 0
source slurm/env.sh
export MASTER_PORT=29908
source slurm/s3_arm.sh rv_mix25 rv data/bench/proc/s3_trainb_300.jsonl,data/bench/proc/s3_rec_train_100.jsonl 1 200
