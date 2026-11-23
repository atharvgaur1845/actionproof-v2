# minutes: 150
# R1 capability retention (PREREGISTRATION_R1.md): RSC s3-4, control, ack-only s0
# after: 1099_r1_smoke
set -uo pipefail
source slurm/claim.sh || exit 0
source slurm/env.sh
source slurm/r1_evalset.sh "rsc_s3=runs/s3_rv_mix_s3 rsc_s4=runs/s3_rv_mix_s4 control_s0=runs/s3_matched_s0 control_s1=runs/s3_matched_s1 control_s2=runs/s3_matched_s2 control_s3=runs/s3_matched_s3 control_s4=runs/s3_matched_s4 ackonly_s0=runs/s3_ackonly_s0"
