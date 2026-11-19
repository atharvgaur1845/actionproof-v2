# minutes: 150
# A12 warm start (SFT on proc_demo demonstrations) and sampled gate (PREREGISTRATION_A12 §4). Writes
# runs/rft_proc/final, results/a12/gate.json and results/a12/h3_predictions.json.
set -uo pipefail
source slurm/claim.sh || exit 0
source slurm/env.sh
.venv/bin/python scripts/freeze_a12.py --check || exit 1
mkdir -p data/rft logs results/a12 results/eval/a12
.venv/bin/python -m actionproof.train.proc_demos --out data/rft/proc_demo_rows.jsonl || exit 1
[ -f runs/rft_proc/final/adapter_config.json ] || .venv/bin/python -m actionproof.train.rft \
  --data data/rft/proc_demo_rows.jsonl --out runs/rft_proc --epochs 2 > logs/600_rft_proc.full.log 2>&1 || { tail -30 logs/600_rft_proc.full.log; exit 1; }
rm -rf runs/rft_proc/checkpoint-*/optimizer.pt
G=results/eval/a12/init__neutral__proc_train_T1x4.jsonl
[ -f ${G%.jsonl}.summary.json ] || .venv/bin/python scripts/eval_proc.py --model CohereLabs/tiny-aya-global \
  --adapter runs/rft_proc/final --split proc_train --n-tasks 100 --samples 4 --temperature 1 --out $G \
  > logs/600_gate.full.log 2>&1 || { tail -30 logs/600_gate.full.log; exit 1; }
.venv/bin/python scripts/a12_gate.py --gate $G
