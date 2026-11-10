# minutes: 260
# after: 370_a5lo_rft_init
set -euo pipefail
source slurm/env.sh
INIT=runs/rft_tiny_a5lo TAG=lo bash slurm/a5_arm.sh a5_state 1
