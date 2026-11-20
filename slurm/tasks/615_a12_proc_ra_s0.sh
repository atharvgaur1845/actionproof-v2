# minutes: 330
# after: 601_a12b_regate
# A12 GRPO proc_ra seed 0 (PREREGISTRATION_A12 §5).
set -uo pipefail
source slurm/claim.sh || exit 0
source slurm/env.sh
source slurm/a12_arm.sh proc_ra 0
