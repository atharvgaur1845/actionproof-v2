# minutes: 200
# S3 addendum 1: class-matched full-episode R_V baseline, seed 0.
set -uo pipefail
source slurm/claim.sh || exit 0
source slurm/env.sh
export MASTER_PORT=29930
.venv/bin/python scripts/freeze_s3a1.py --check || exit 1
source slurm/s3_arm.sh matched rv data/bench/proc/s3_trainb_half.jsonl,data/bench/proc/s3_matched_full_half.jsonl 0 200
