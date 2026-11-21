# minutes: 240
# Stage 4 development: FCA seed 1, 200 steps, RS pool (kill criteria in docs/research/algorithm/FCA_GRPO.md §7).
set -uo pipefail
source slurm/claim.sh || exit 0
source slurm/env.sh
export MASTER_PORT=29941
source slurm/s4_arm.sh fca data/bench/proc/s3_trainb_half.jsonl,data/bench/proc/s3_rec_train_half.jsonl 1 200 4
