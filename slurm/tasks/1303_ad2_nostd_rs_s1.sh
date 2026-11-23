# minutes: 330
# after: 1302_ad2_nostd_rs_s0
# V2-E1 (docs/research/algorithm_discovery_v2/V2E1_CONDITION_B.md): ad2_nostd_rs_s1, condition B (resampled worlds).
set -uo pipefail
source slurm/claim.sh || exit 0
source slurm/env.sh
export MASTER_PORT=29303
source slurm/ad2_arm.sh V2E1_CONDITION_B nostd resampled 1 nostd_rs
