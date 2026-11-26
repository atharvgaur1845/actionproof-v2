# minutes: 30
# Reproducibility check of the analyses added to scripts/verify_numbers.sh (P4, P5, TXN): rerun each from the stored
# episodes and compare with the committed outputs (sha256 list written from the git copy). CPU only.
set -uo pipefail
source slurm/claim.sh || exit 0
source slurm/env.sh
PY=.venv/bin/python
TXN_TAGS=A1,A2,A3,A4,A5,A6,A_s0,A_s1,A_s2,B0,B1,B2,B3,B4,D_s0,D_s1,D_s2,Dn_s0,Dn_s1,Dn_s2,Dw,W,aya32b_bf16,aya8b,tag_base
$PY scripts/p4_stats.py >/dev/null || exit 1
$PY scripts/p5_stats.py >/dev/null || exit 1
$PY scripts/txn_stats.py --split txn_test --tags "$TXN_TAGS" >/dev/null || exit 1
$PY scripts/txn_invariants_recheck.py --split txn_test >/dev/null || exit 1
$PY scripts/txn_stats.py --split txn_test --out results/txn/stats_all_descriptive__txn_test.json --pairs Dwaug_s0:Dw,Dwaug_s1:Dw,Dwaug_s2:Dw >/dev/null || exit 1
$PY scripts/txn_compact.py results/txn/stats_all_descriptive__txn_test.json results/txn/compact_descriptive__txn_test.json >/dev/null || exit 1
sha256sum -c slurm/tasks/1730b_expected.sha256 || exit 1
echo VERIFY_NEW_OK
