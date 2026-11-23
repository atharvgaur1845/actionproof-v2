# minutes: 240
# after: 1009_c5g_smoke
# Stage 5 development (docs/research/algorithm/C5G_DEV.md): guide seed 0, 200 steps, RSC pool.
set -uo pipefail
source slurm/claim.sh || exit 0
source slurm/env.sh
export MASTER_PORT=29911
source slurm/s5_arm.sh guide data/bench/proc/s3_trainb_half.jsonl,data/bench/proc/s3_rec_train_half.jsonl 0 200
