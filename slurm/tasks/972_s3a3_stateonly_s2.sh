# minutes: 200
# S3 addendum 3 (PREREGISTRATION_S3_ADDENDUM3): state-only recovery starts, seed 2.
set -uo pipefail
source slurm/claim.sh || exit 0
source slurm/env.sh
export MASTER_PORT=29972
.venv/bin/python scripts/freeze_s3a3.py --check || exit 1
source slurm/s3v_arm.sh stateonly --hide-prefix data/bench/proc/s3_trainb_half.jsonl,data/bench/proc/s3_rec_train_half.jsonl 2 200
