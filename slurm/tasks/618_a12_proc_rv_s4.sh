# minutes: 330
# after: 601_a12b_regate
# A12 GRPO proc_rv seed 4 (PREREGISTRATION_A12 §5).
# Seeds 3-4 start only if they can finish, with their confirmatory runs, by 2026-11-23 12:00 IST (§5).
set -uo pipefail
[ $(( $(date +%s) + 7*3600 )) -le $(date -d "2026-11-23 12:00 IST" +%s) ] || { echo "too late for this seed (registered rule)"; rm -f "$0"; exit 0; }
source slurm/claim.sh || exit 0
source slurm/env.sh
source slurm/a12_arm.sh proc_rv 4
