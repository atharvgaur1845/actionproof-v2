# minutes: 150
# P6 preregistered RL continuation; paired by the Dw_aug seed.
set -euo pipefail
source slurm/claim.sh || exit 0
source slurm/env.sh
source slurm/p6_train_common.sh
run_p6_train surface 0
