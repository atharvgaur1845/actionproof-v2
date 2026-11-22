# minutes: 150
# E1/E2 (PREREGISTRATION_E1E2): class-matched control runs
set -uo pipefail
source slurm/claim.sh || exit 0
source slurm/env.sh
source slurm/e1e2_set.sh "s3_matched_s0=runs/s3_matched_s0/final s3_matched_s1=runs/s3_matched_s1/final s3_matched_s2=runs/s3_matched_s2/final s3_matched_s3=runs/s3_matched_s3/final s3_matched_s4=runs/s3_matched_s4/final "
