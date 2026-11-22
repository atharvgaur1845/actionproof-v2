# minutes: 150
# E1/E2 (PREREGISTRATION_E1E2): warm start and A12 R_V
set -uo pipefail
source slurm/claim.sh || exit 0
source slurm/env.sh
source slurm/e1e2_set.sh "warm_start=  a12_rv_s0=runs/a12_proc_rv_s0/final a12_rv_s1=runs/a12_proc_rv_s1/final a12_rv_s2=runs/a12_proc_rv_s2/final a12_rv_s3=runs/a12_proc_rv_s3/final a12_rv_s4=runs/a12_proc_rv_s4/final"
