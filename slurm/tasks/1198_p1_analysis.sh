# minutes: 30
# P1 registered analysis (scripts/p1_stats.py, frozen) + descriptive failure-stage diagnosis; runs unattended when every P1 task is done
# after: 1150_p1_c_s0 1151_p1_c_s1 1152_p1_c_s2 1153_p1_dwarm 1154_p1_d_s0 1155_p1_d_s1 1156_p1_d_s2 1157_p1_dev_ab 1160_p1_test_A_s0 1161_p1_test_B_s0 1162_p1_test_C_s0 1163_p1_test_D_s0 1164_p1_test_A_s1 1165_p1_test_B_s1 1166_p1_test_C_s1 1167_p1_test_D_s1 1168_p1_test_A_s2 1169_p1_test_B_s2 1170_p1_test_C_s2 1171_p1_test_D_s2 1172_p1_auditkit 
set -uo pipefail
source slurm/claim.sh || exit 0
source slurm/env.sh
.venv/bin/python scripts/freeze_p1.py --check || exit 1
.venv/bin/python scripts/p1_stats.py > logs/p1_stats.log 2>&1 || { tail -30 logs/p1_stats.log; exit 1; }
cat logs/p1_stats.log
mkdir -p results/quality
for ph in dev test; do .venv/bin/python scripts/diagnose_failures.py --glob "results/p1/$ph/*__*.jsonl" --out results/quality/diagnosis_p1_$ph.json > /dev/null; done
.venv/bin/python scripts/diagnosis_report.py
