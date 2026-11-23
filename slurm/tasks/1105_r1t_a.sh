# minutes: 300
# R1-T τ-bench (PREREGISTRATION_R1_TAU.md): base, warm start, A12 R_V s0-2
# after: 1104_r1t_pilot
set -uo pipefail
source slurm/claim.sh || exit 0
source slurm/env.sh
source slurm/r1_tau_set.sh "base= warm_start= a12_rv_s0=runs/a12_proc_rv_s0 a12_rv_s1=runs/a12_proc_rv_s1 a12_rv_s2=runs/a12_proc_rv_s2"
