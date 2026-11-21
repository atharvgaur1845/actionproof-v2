# minutes: 150
# after: 960_s3a2_matched_s3 961_s3a2_matched_s4
# S3 addendum 2 Part A: control seeds 3-4 once on every S3 confirmatory split.
set -uo pipefail
source slurm/claim.sh || exit 0
source slurm/env.sh
export MASTER_PORT=29966
.venv/bin/python scripts/freeze_s3a2.py --check || exit 1
source slurm/s3_conf.sh "s3_matched_s3=runs/s3_matched_s3/final s3_matched_s4=runs/s3_matched_s4/final"
