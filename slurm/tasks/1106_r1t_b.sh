# minutes: 240
# R1-T τ-bench (PREREGISTRATION_R1_TAU.md): RSC s0-2
# after: 1104_r1t_pilot
set -uo pipefail
source slurm/claim.sh || exit 0
source slurm/env.sh
source slurm/r1_tau_set.sh "rsc_s0=runs/s3_rv_mix_s0 rsc_s1=runs/s3_rv_mix_s1 rsc_s2=runs/s3_rv_mix_s2"
