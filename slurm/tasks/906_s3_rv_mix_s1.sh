# minutes: 200
# Stage 3 development stage B (200 steps; dev splits only): F, 50%% recovery starts, seed 1
set -uo pipefail
source slurm/claim.sh || exit 0
source slurm/env.sh
export MASTER_PORT=29906
source slurm/s3_arm.sh rv_mix rv data/bench/proc/s3_trainb_half.jsonl,data/bench/proc/s3_rec_train_half.jsonl 1 200
