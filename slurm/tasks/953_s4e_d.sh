# minutes: 200
# S4-EVAL: class-matched control seeds 0-2
set -uo pipefail
source slurm/claim.sh || exit 0
source slurm/env.sh
export MASTER_PORT=29953
source slurm/s4_evalset.sh "s3_matched_s0=runs/s3_matched_s0/final s3_matched_s1=runs/s3_matched_s1/final s3_matched_s2=runs/s3_matched_s2/final"
