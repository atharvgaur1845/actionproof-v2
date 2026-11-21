# minutes: 120
# after: 960_s3a2_matched_s3 961_s3a2_matched_s4 962_s3a2_rb_mix_s0 963_s3a2_rb_mix_s1 964_s3a2_rb_matched_s0 965_s3a2_rb_matched_s1
# S3 addendum 2 Part B: new models once on the targeted partial-payment splits.
set -uo pipefail
source slurm/claim.sh || exit 0
source slurm/env.sh
export MASTER_PORT=29968
.venv/bin/python scripts/freeze_s3a2.py --check || exit 1
source slurm/s3_ppset.sh "s3_matched_s3=runs/s3_matched_s3/final s3_matched_s4=runs/s3_matched_s4/final s3_rb_mix_s0=runs/s3_rb_mix_s0/final s3_rb_mix_s1=runs/s3_rb_mix_s1/final s3_rb_matched_s0=runs/s3_rb_matched_s0/final s3_rb_matched_s1=runs/s3_rb_matched_s1/final"
