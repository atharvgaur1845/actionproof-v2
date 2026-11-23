# minutes: 150
# R1 capability retention (PREREGISTRATION_R1.md): base, warm starts, A12 R_V
# after: 1099_r1_smoke
set -uo pipefail
source slurm/claim.sh || exit 0
source slurm/env.sh
source slurm/r1_evalset.sh "base= warm_start= s1_warm_start= a12_rv_s0=runs/a12_proc_rv_s0 a12_rv_s1=runs/a12_proc_rv_s1 a12_rv_s2=runs/a12_proc_rv_s2 a12_rv_s3=runs/a12_proc_rv_s3 a12_rv_s4=runs/a12_proc_rv_s4 "
