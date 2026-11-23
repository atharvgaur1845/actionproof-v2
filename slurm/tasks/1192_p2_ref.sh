# minutes: 90
# P2 (PREREGISTRATION_P2.md): references on the test union: warm start W, class-matched control M (three seeds)
set -uo pipefail
source slurm/claim.sh || exit 0
source slurm/env.sh
.venv/bin/python scripts/freeze_p2.py --check || exit 1
source slurm/p2_evalset.sh ref "W|runs/rft_proc/merged|-|-" "p1_test_full p1_test_pp p1_test_rec p1_test_wf4 p1_test_combo"
source slurm/p2_evalset.sh ref "M_s0|runs/rft_proc/merged|runs/s3_matched_s0/final|-" "p1_test_full p1_test_pp p1_test_rec p1_test_wf4 p1_test_combo"
source slurm/p2_evalset.sh ref "M_s1|runs/rft_proc/merged|runs/s3_matched_s1/final|-" "p1_test_full p1_test_pp p1_test_rec p1_test_wf4 p1_test_combo"
source slurm/p2_evalset.sh ref "M_s2|runs/rft_proc/merged|runs/s3_matched_s2/final|-" "p1_test_full p1_test_pp p1_test_rec p1_test_wf4 p1_test_combo"
