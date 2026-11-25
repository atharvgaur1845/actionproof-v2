# minutes: 120
# Descriptive (research_audit/NOTE_DWARM_EVAL.md): evaluate P1's recovery-SFT warm start p1_dwarm (never evaluated) on P1 dev + frozen test.
set -uo pipefail
source slurm/claim.sh || exit 0
source slurm/env.sh
source slurm/p1_evalset.sh dev "Dw|runs/p1_dwarm/merged|-|0" "p1_dev_full p1_dev_rec"
source slurm/p1_evalset.sh test "Dw|runs/p1_dwarm/merged|-|0" "p1_test_full p1_test_pp p1_test_rec p1_test_wf4 p1_test_combo"
echo P1_DWARM_EVAL_OK
