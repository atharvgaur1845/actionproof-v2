# minutes: 200
# S4-EVAL: warm start, R_V seeds 0-2
set -uo pipefail
source slurm/claim.sh || exit 0
source slurm/env.sh
export MASTER_PORT=29950
source slurm/s4_evalset.sh "warm_start= a12_rv_s0=runs/a12_proc_rv_s0/final a12_rv_s1=runs/a12_proc_rv_s1/final a12_rv_s2=runs/a12_proc_rv_s2/final"
