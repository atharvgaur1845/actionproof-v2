# minutes: 300
# after: 410_a7_init
# Amendment 7 E7-3 (PREREGISTRATION_A7 §1): pcstr_d_noreport on rft_tiny_a7, seed 0.
set -euo pipefail
source slurm/env.sh
bash slurm/a7_arm.sh pcstr_d_noreport 0 rft_tiny_a7 a7
