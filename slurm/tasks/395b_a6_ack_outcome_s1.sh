# minutes: 300
# after: 390_a6_init
# Amendment 6 arm ack_outcome, seed 1 (PREREGISTRATION_A6 §2).
set -euo pipefail
source slurm/env.sh
bash slurm/a6_arm.sh ack_outcome 1
