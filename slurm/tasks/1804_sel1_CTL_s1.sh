# minutes: 150
# SEL1 (PREREGISTRATION_SEL1.md): arm CTL seed 1 -- new LoRA (actionproof.train.rft, 1 epoch, seed 1) on the parent
# Dw_aug seed 1 (runs/p4_dwaug_s1/merged), then the registered evaluations (txn_test, sel1_test; 1 greedy + 2 sampled).
# after: 1800_sel1_build
set -uo pipefail
source slurm/claim.sh || exit 0
source slurm/env.sh
.venv/bin/python scripts/freeze_sel1.py --check || exit 1
mkdir -p logs results/sel1
D=data/rft/sel1_ctl_rows.jsonl
[ -f runs/sel1_CTL_s1/final/adapter_config.json ] || .venv/bin/python -m actionproof.train.rft --model runs/p4_dwaug_s1/merged --data $D --out runs/sel1_CTL_s1 --epochs 1 --seed 1 > logs/sel1_CTL_s1.full.log 2>&1 || { tail -30 logs/sel1_CTL_s1.full.log; exit 1; }
rm -rf runs/sel1_CTL_s1/checkpoint-*/optimizer.pt
grep -o "train_runtime[^,]*" logs/sel1_CTL_s1.full.log | tail -1
for sp in sel1_test txn_test; do
  [ -f results/sel1/CTL_s1__$sp.summary.json ] && { echo "skip $sp"; continue; }
  rm -f results/sel1/CTL_s1__$sp.jsonl
  .venv/bin/python scripts/sel1_eval.py --model runs/p4_dwaug_s1/merged --adapter runs/sel1_CTL_s1/final --tag CTL_s1 --split $sp --samples 2 > logs/sel1_CTL_s1_$sp.log 2>&1 || { tail -30 logs/sel1_CTL_s1_$sp.log; exit 1; }
  tail -1 logs/sel1_CTL_s1_$sp.log
done
echo SEL1_CTL_s1_OK
