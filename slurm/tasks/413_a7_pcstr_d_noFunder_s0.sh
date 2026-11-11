# minutes: 300
# Amendment 7 E7-4 (PREREGISTRATION_A7 §2): M2c pcstr_d_noFunder (ND kept, F_under removed) on rft_tiny_a6, seed 0.
set -euo pipefail
source slurm/env.sh
bash slurm/a7_arm.sh pcstr_d_noFunder 0 rft_tiny_a6 a6
