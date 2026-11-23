# minutes: 300
# after: 1210_ad_baselines
# Algorithm discovery AD-D1 (docs/research/algorithm_discovery/EXPERIMENT_PROTOCOL.md): ad_grpo_s0.
set -uo pipefail
source slurm/claim.sh || exit 0
source slurm/env.sh
export MASTER_PORT=29202
source slurm/ad_arm.sh grpo 0 200
