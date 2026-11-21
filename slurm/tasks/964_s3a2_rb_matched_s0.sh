# minutes: 200
# S3 addendum 2 Part B: rebalanced class-matched control, seed 0.
set -uo pipefail
source slurm/claim.sh || exit 0
source slurm/env.sh
export MASTER_PORT=29964
.venv/bin/python scripts/freeze_s3a2.py --check || exit 1
source slurm/s3_arm.sh rb_matched rv data/bench/proc/s3_trainb_half_rb.jsonl,data/bench/proc/s3_matched_full_half_rb.jsonl 0 200
