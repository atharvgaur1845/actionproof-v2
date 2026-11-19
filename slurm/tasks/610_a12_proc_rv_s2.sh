# minutes: 330
# after: 600_a12_warmstart
# A12 GRPO proc_rv seed 2 (PREREGISTRATION_A12 §5).
set -uo pipefail
source slurm/claim.sh || exit 0
source slurm/env.sh
source slurm/a12_arm.sh proc_rv 2
