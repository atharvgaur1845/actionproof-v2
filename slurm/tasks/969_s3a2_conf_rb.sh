# minutes: 150
# after: 962_s3a2_rb_mix_s0 963_s3a2_rb_mix_s1 964_s3a2_rb_matched_s0 965_s3a2_rb_matched_s1
# S3 addendum 2 Part B (secondary): rebalanced models once on every S3 confirmatory split.
set -uo pipefail
source slurm/claim.sh || exit 0
source slurm/env.sh
export MASTER_PORT=29969
.venv/bin/python scripts/freeze_s3a2.py --check || exit 1
source slurm/s3_conf.sh "s3_rb_mix_s0=runs/s3_rb_mix_s0/final s3_rb_mix_s1=runs/s3_rb_mix_s1/final s3_rb_matched_s0=runs/s3_rb_matched_s0/final s3_rb_matched_s1=runs/s3_rb_matched_s1/final"
