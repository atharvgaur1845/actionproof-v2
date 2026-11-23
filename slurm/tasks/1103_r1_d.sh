# minutes: 150
# R1 capability retention (PREREGISTRATION_R1.md): ack-only s1-4, SR1, Stage 1 chain
# after: 1099_r1_smoke
set -uo pipefail
source slurm/claim.sh || exit 0
source slurm/env.sh
source slurm/r1_evalset.sh "ackonly_s1=runs/s3_ackonly_s1 ackonly_s2=runs/s3_ackonly_s2 ackonly_s3=runs/s3_ackonly_s3 ackonly_s4=runs/s3_ackonly_s4 sr1_s0=runs/s3_sr_mix_s0 sr1_s1=runs/s3_sr_mix_s1 sr1_s2=runs/s3_sr_mix_s2 s1_pcstr_d_s0=runs/grpo_tiny_v2_pcstr_d_s0 s1_pcstr_d_s1=runs/grpo_tiny_v2_pcstr_d_s1 s1_tool_trust_s0=runs/grpo_tiny_v2_tool_trust_s0 s1_tool_trust_s1=runs/grpo_tiny_v2_tool_trust_s1"
