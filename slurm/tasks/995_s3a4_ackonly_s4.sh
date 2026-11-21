# minutes: 200
# S3 addendum 4 (PREREGISTRATION_S3_ADDENDUM4): acknowledgement-only recovery starts, seed 4.
set -uo pipefail
source slurm/claim.sh || exit 0
source slurm/env.sh
export MASTER_PORT=29995
.venv/bin/python scripts/freeze_s3a4.py --check || exit 1
source slurm/s3_arm.sh ackonly rv data/bench/proc/s3_trainb_half.jsonl,data/bench/proc/s3_rec_train_half_ackonly.jsonl 4 200
