# minutes: 300
# after: 390_a6_init
# Amendment 6 arm pcstr_d, seed 1 (PREREGISTRATION_A6 §2).
set -euo pipefail
source slurm/env.sh
bash slurm/a6_arm.sh pcstr_d 1
