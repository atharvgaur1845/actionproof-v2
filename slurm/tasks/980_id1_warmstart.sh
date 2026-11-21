# minutes: 120
# ID1 (PREREGISTRATION_ID1): warm start with no verifying demonstrations, its sampled gate and proc_conf evaluation.
set -uo pipefail
source slurm/claim.sh || exit 0
source slurm/env.sh
.venv/bin/python scripts/freeze_id1.py --check || exit 1
mkdir -p data/rft logs results/id1
.venv/bin/python -m actionproof.train.proc_demos --verify-share 0 --out data/rft/proc_demo_rows_v0.jsonl || exit 1
[ -f runs/rft_proc_v0/final/adapter_config.json ] || .venv/bin/python -m actionproof.train.rft \
  --data data/rft/proc_demo_rows_v0.jsonl --out runs/rft_proc_v0 --epochs 2 > logs/980_rft_v0.full.log 2>&1 \
  || { tail -30 logs/980_rft_v0.full.log; exit 1; }
rm -rf runs/rft_proc_v0/checkpoint-*/optimizer.pt
[ -f runs/rft_proc_v0/merged/config.json ] || .venv/bin/python -m actionproof.train.merge runs/rft_proc_v0/final \
  CohereLabs/tiny-aya-global > logs/980_merge.log 2>&1 || { tail -20 logs/980_merge.log; exit 1; }
G=results/id1/warm_v0__neutral__proc_train_T1x4.jsonl
[ -f ${G%.jsonl}.summary.json ] || .venv/bin/python scripts/eval_proc.py --model runs/rft_proc_v0/merged \
  --split proc_train --n-tasks 100 --samples 4 --temperature 1 --out $G > logs/980_gate.log 2>&1 || { tail -20 logs/980_gate.log; exit 1; }
C=results/id1/warm_v0__neutral__proc_conf.jsonl
[ -f ${C%.jsonl}.summary.json ] || .venv/bin/python scripts/eval_proc.py --model runs/rft_proc_v0/merged \
  --split proc_conf --out $C > logs/980_conf.log 2>&1 || { tail -20 logs/980_conf.log; exit 1; }
tail -1 logs/980_conf.log
