# minutes: 120
# after: 915_s3_matched_s0 916_s3_matched_s1 917_s3_matched_s2
# S3 addendum 1: class-matched baseline evaluation on every S3 confirmatory split.
set -uo pipefail
source slurm/claim.sh || exit 0
source slurm/env.sh
export MASTER_PORT=29924
.venv/bin/python scripts/freeze_s3a1.py --check || exit 1
source slurm/s3_conf.sh "s3_matched_s0=runs/s3_matched_s0/final s3_matched_s1=runs/s3_matched_s1/final s3_matched_s2=runs/s3_matched_s2/final"
