# minutes: 200
# S3 addendum 2 Part A: class-matched control, seed 3.
set -uo pipefail
source slurm/claim.sh || exit 0
source slurm/env.sh
export MASTER_PORT=29960
.venv/bin/python scripts/freeze_s3a2.py --check || exit 1
source slurm/s3_arm.sh matched rv data/bench/proc/s3_trainb_half.jsonl,data/bench/proc/s3_matched_full_half.jsonl 3 200
