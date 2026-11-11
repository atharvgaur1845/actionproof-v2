# minutes: 300
# after: 410_a7_init
# Amendment 7 E7-5 (conditional on H7-1 at seed 0; keep in hold until then): pcstr_d_noreport on rft_tiny_a7, seed 1.
set -euo pipefail
source slurm/env.sh
bash slurm/a7_arm.sh pcstr_d_noreport 1 rft_tiny_a7 a7
