# minutes: 330
# after: 1013_c5g_guide_s1
# V2-E1 (docs/research/algorithm_discovery_v2/V2E1_CONDITION_B.md): ad2_grpo_rs_s0, condition B (resampled worlds).
set -uo pipefail
source slurm/claim.sh || exit 0
source slurm/env.sh
export MASTER_PORT=29300
source slurm/ad2_arm.sh V2E1_CONDITION_B grpo resampled 0 grpo_rs
