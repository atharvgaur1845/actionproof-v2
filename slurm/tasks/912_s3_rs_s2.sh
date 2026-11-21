# minutes: 200
# S3 confirmatory training: RS-GRPO seed 2 (PREREGISTRATION_S3 §2).
set -uo pipefail
source slurm/claim.sh || exit 0
source slurm/env.sh
export MASTER_PORT=29912
.venv/bin/python scripts/freeze_s3.py --check || exit 1
source slurm/s3_arm.sh rv_mix rv data/bench/proc/s3_trainb_half.jsonl,data/bench/proc/s3_rec_train_half.jsonl 2 200
