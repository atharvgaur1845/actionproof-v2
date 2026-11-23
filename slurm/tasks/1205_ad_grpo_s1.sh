# minutes: 300
# after: 1203_ad_nostd_s0
# Algorithm discovery AD-D1 (docs/research/algorithm_discovery/EXPERIMENT_PROTOCOL.md): ad_grpo_s1.
set -uo pipefail
source slurm/claim.sh || exit 0
source slurm/env.sh
export MASTER_PORT=29205
source slurm/ad_arm.sh grpo 1 200
