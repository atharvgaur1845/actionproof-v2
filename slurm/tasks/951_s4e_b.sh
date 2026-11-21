# minutes: 200
# S4-EVAL: R_V seeds 3-4, RSC seeds 0-1
set -uo pipefail
source slurm/claim.sh || exit 0
source slurm/env.sh
export MASTER_PORT=29951
source slurm/s4_evalset.sh "a12_rv_s3=runs/a12_proc_rv_s3/final a12_rv_s4=runs/a12_proc_rv_s4/final s3_rv_mix_s0=runs/s3_rv_mix_s0/final s3_rv_mix_s1=runs/s3_rv_mix_s1/final"
