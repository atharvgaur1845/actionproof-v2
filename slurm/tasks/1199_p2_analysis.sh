# minutes: 30
# P2 registered analysis (scripts/p2_stats.py, frozen; coverage gate first); runs unattended when every P2 task is done
# after: 1180_p2_smoke 1181_p2_oracle 1182_p2_ewarm 1183_p2_e_s0 1184_p2_e_s1 1185_p2_e_s2 1186_p2_dev_b2g 1187_p2_test_B2 1188_p2_test_Ag 1189_p2_test_Cg 1190_p2_test_Dg 1191_p2_test_E 1192_p2_ref 1193_p2_shift_A 1194_p2_shift_C 1195_p2_shift_D 1196_p2_shift_E  1198_p1_analysis
set -uo pipefail
source slurm/claim.sh || exit 0
source slurm/env.sh
.venv/bin/python scripts/freeze_p2.py --check || exit 1
.venv/bin/python scripts/p2_stats.py > logs/p2_stats.log 2>&1; rc=$?
cat logs/p2_stats.log
mkdir -p results/quality
.venv/bin/python scripts/diagnose_failures.py --glob "results/p2/test/*__*.jsonl" --out results/quality/diagnosis_p2_test.json > /dev/null
exit $rc
