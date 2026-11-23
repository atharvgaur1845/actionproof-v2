# minutes: 150
# P2 (PREREGISTRATION_P2.md): frozen test once: E and Eg, three seeds
# after: 1183_p2_e_s0 1184_p2_e_s1 1185_p2_e_s2
set -uo pipefail
source slurm/claim.sh || exit 0
source slurm/env.sh
.venv/bin/python scripts/freeze_p2.py --check || exit 1
source slurm/p2_evalset.sh test "E_s0|runs/p2_ewarm/merged|runs/p2_e_s0/final|L" "p1_test_full p1_test_pp p1_test_rec p1_test_wf4 p1_test_combo"
source slurm/p2_evalset.sh test "Eg_s0|runs/p2_ewarm/merged|runs/p2_e_s0/final|LG" "p1_test_full p1_test_pp p1_test_rec p1_test_wf4 p1_test_combo"
source slurm/p2_evalset.sh test "E_s1|runs/p2_ewarm/merged|runs/p2_e_s1/final|L" "p1_test_full p1_test_pp p1_test_rec p1_test_wf4 p1_test_combo"
source slurm/p2_evalset.sh test "Eg_s1|runs/p2_ewarm/merged|runs/p2_e_s1/final|LG" "p1_test_full p1_test_pp p1_test_rec p1_test_wf4 p1_test_combo"
source slurm/p2_evalset.sh test "E_s2|runs/p2_ewarm/merged|runs/p2_e_s2/final|L" "p1_test_full p1_test_pp p1_test_rec p1_test_wf4 p1_test_combo"
source slurm/p2_evalset.sh test "Eg_s2|runs/p2_ewarm/merged|runs/p2_e_s2/final|LG" "p1_test_full p1_test_pp p1_test_rec p1_test_wf4 p1_test_combo"
