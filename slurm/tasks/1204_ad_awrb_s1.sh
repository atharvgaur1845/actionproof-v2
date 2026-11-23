# minutes: 300
# after: 1202_ad_grpo_s0
# Algorithm discovery AD-D1 (docs/research/algorithm_discovery/EXPERIMENT_PROTOCOL.md): ad_awrb_s1.
set -uo pipefail
source slurm/claim.sh || exit 0
source slurm/env.sh
export MASTER_PORT=29204
source slurm/ad_arm.sh awrb 1 200
