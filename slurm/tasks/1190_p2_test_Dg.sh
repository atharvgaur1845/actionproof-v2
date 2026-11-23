# minutes: 120
# P2 (PREREGISTRATION_P2.md): frozen test once: Dg, three seeds
# after: 1154_p1_d_s0 1155_p1_d_s1 1156_p1_d_s2
set -uo pipefail
source slurm/claim.sh || exit 0
source slurm/env.sh
.venv/bin/python scripts/freeze_p2.py --check || exit 1
[ -f runs/s3_rv_mix_s0/merged/config.json ] || .venv/bin/python -m actionproof.train.merge runs/s3_rv_mix_s0/final runs/rft_proc/merged > logs/p2_merge_rsc0.log 2>&1 || exit 1
[ -f runs/s3_rv_mix_s1/merged/config.json ] || .venv/bin/python -m actionproof.train.merge runs/s3_rv_mix_s1/final runs/rft_proc/merged > logs/p2_merge_rsc1.log 2>&1 || exit 1
[ -f runs/s3_rv_mix_s2/merged/config.json ] || .venv/bin/python -m actionproof.train.merge runs/s3_rv_mix_s2/final runs/rft_proc/merged > logs/p2_merge_rsc2.log 2>&1 || exit 1
source slurm/p2_evalset.sh test "Dg_s0|runs/p1_dwarm/merged|runs/p1_d_s0/final|G" "p1_test_full p1_test_pp p1_test_rec p1_test_wf4 p1_test_combo"
source slurm/p2_evalset.sh test "Dg_s1|runs/p1_dwarm/merged|runs/p1_d_s1/final|G" "p1_test_full p1_test_pp p1_test_rec p1_test_wf4 p1_test_combo"
source slurm/p2_evalset.sh test "Dg_s2|runs/p1_dwarm/merged|runs/p1_d_s2/final|G" "p1_test_full p1_test_pp p1_test_rec p1_test_wf4 p1_test_combo"
