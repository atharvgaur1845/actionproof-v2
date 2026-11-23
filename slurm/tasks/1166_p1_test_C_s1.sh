# minutes: 150
# P1 practical agent (PREREGISTRATION_P1.md): frozen test, arm C seed 1, five test splits once
# after: 1151_p1_c_s1
set -uo pipefail
source slurm/claim.sh || exit 0
source slurm/env.sh
.venv/bin/python scripts/freeze_p1.py --check || exit 1
source slurm/p1_evalset.sh test "C_s1|runs/s3_rv_mix_s1/merged|runs/p1_c_s1/final|0" "p1_test_full p1_test_pp p1_test_rec p1_test_wf4 p1_test_combo"
