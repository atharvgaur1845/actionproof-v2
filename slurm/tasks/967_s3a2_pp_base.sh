# minutes: 150
# S3 addendum 2 Part B: existing models once on the targeted partial-payment splits.
set -uo pipefail
source slurm/claim.sh || exit 0
source slurm/env.sh
export MASTER_PORT=29967
.venv/bin/python scripts/freeze_s3a2.py --check || exit 1
source slurm/s3_ppset.sh "warm_start= a12_rv_s0=runs/a12_proc_rv_s0/final a12_rv_s1=runs/a12_proc_rv_s1/final a12_rv_s2=runs/a12_proc_rv_s2/final a12_rv_s3=runs/a12_proc_rv_s3/final a12_rv_s4=runs/a12_proc_rv_s4/final s3_rv_mix_s0=runs/s3_rv_mix_s0/final s3_rv_mix_s1=runs/s3_rv_mix_s1/final s3_rv_mix_s2=runs/s3_rv_mix_s2/final s3_rv_mix_s3=runs/s3_rv_mix_s3/final s3_rv_mix_s4=runs/s3_rv_mix_s4/final s3_matched_s0=runs/s3_matched_s0/final s3_matched_s1=runs/s3_matched_s1/final s3_matched_s2=runs/s3_matched_s2/final"
