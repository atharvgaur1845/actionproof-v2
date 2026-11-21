# minutes: 150
# after: 991_s3a4_ackonly_s0 992_s3a4_ackonly_s1 993_s3a4_ackonly_s2 994_s3a4_ackonly_s3 995_s3a4_ackonly_s4
# S3 addendum 4: acknowledgement-only models once on the S3 confirmatory splits.
set -uo pipefail
source slurm/claim.sh || exit 0
source slurm/env.sh
export MASTER_PORT=29996
.venv/bin/python scripts/freeze_s3a4.py --check || exit 1
source slurm/s3_conf.sh "s3_ackonly_s0=runs/s3_ackonly_s0/final s3_ackonly_s1=runs/s3_ackonly_s1/final s3_ackonly_s2=runs/s3_ackonly_s2/final s3_ackonly_s3=runs/s3_ackonly_s3/final s3_ackonly_s4=runs/s3_ackonly_s4/final"
