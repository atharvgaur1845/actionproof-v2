# minutes: 15
# after: 331_a4_h11_sampled 332_a4_gate
# B' pre-training gates (PREREGISTRATION_A5 §4). Exit != 0 blocks every dependent GPU training task.
set -euo pipefail  # pipefail: a pytest failure fails the pipeline
source slurm/env.sh
.venv/bin/python scripts/freeze_bench.py --check
.venv/bin/python scripts/freeze_a4.py --check
.venv/bin/python scripts/freeze_a5.py --check
.venv/bin/python -m pytest -q tests/ 2>&1 | tail -1   # set -e: any failure exits non-zero
for f in results/eval/a4/h11/{init,tool_trust_s0,outcome_s0,pcstr_d_s0}__val_T1x8.summary.json results/eval/a4/gate/init__a4_train_B_T1x4.summary.json; do
  [ -f $f ] || { echo "GATE FAIL: S5 output missing $f"; exit 1; }
done
echo "GATES PASS"
