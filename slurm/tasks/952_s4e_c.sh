# minutes: 200
# S4-EVAL: RSC seeds 2-4
set -uo pipefail
source slurm/claim.sh || exit 0
source slurm/env.sh
export MASTER_PORT=29952
source slurm/s4_evalset.sh "s3_rv_mix_s2=runs/s3_rv_mix_s2/final s3_rv_mix_s3=runs/s3_rv_mix_s3/final s3_rv_mix_s4=runs/s3_rv_mix_s4/final"
