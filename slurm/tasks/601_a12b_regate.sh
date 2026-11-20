# minutes: 40
# A12 addendum 1: gate at the training group size (8 samples per task) on proc_train_b, same warm start.
set -uo pipefail
source slurm/claim.sh || exit 0
source slurm/env.sh
.venv/bin/python scripts/freeze_a12.py --check || exit 1
.venv/bin/python scripts/freeze_a12b.py --check || exit 1
mkdir -p results/a12b results/eval/a12
G=results/eval/a12/init__neutral__proc_train_b_T1x8.jsonl
[ -f ${G%.jsonl}.summary.json ] || .venv/bin/python scripts/eval_proc.py --model CohereLabs/tiny-aya-global \
  --adapter runs/rft_proc/final --split proc_train_b --n-tasks 100 --samples 8 --temperature 1 --out $G \
  > logs/601_gate.full.log 2>&1 || { tail -30 logs/601_gate.full.log; exit 1; }
.venv/bin/python scripts/a12_gate.py --gate $G --out results/a12b/gate.json
