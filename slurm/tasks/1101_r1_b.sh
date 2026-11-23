# minutes: 150
# R1 capability retention (PREREGISTRATION_R1.md): A12 R_A, RSC s0-2
# after: 1099_r1_smoke
set -uo pipefail
source slurm/claim.sh || exit 0
source slurm/env.sh
source slurm/r1_evalset.sh "a12_ra_s0=runs/a12_proc_ra_s0 a12_ra_s1=runs/a12_proc_ra_s1 a12_ra_s2=runs/a12_proc_ra_s2 a12_ra_s3=runs/a12_proc_ra_s3 a12_ra_s4=runs/a12_proc_ra_s4 rsc_s0=runs/s3_rv_mix_s0 rsc_s1=runs/s3_rv_mix_s1 rsc_s2=runs/s3_rv_mix_s2 "
