# minutes: 240
# P4 (PREREGISTRATION_P4.md): arm Dwaug_s0 — SFT (actionproof.train.rft, 2 epochs, seed 0) then P4 evaluation.
set -uo pipefail
source slurm/claim.sh || exit 0
source slurm/env.sh
.venv/bin/python scripts/freeze_p4.py --check || exit 1
mkdir -p logs results/p4
[ -f runs/p4_dwaug_s0/final/adapter_config.json ] || .venv/bin/python -m actionproof.train.rft --data data/rft/p4_aug_rows.jsonl --out runs/p4_dwaug_s0 --epochs 2 --seed 0 > logs/p4_Dwaug_s0.full.log 2>&1 || { tail -30 logs/p4_Dwaug_s0.full.log; exit 1; }
rm -rf runs/p4_dwaug_s0/checkpoint-*/optimizer.pt
[ -f runs/p4_dwaug_s0/merged/config.json ] || .venv/bin/python -m actionproof.train.merge runs/p4_dwaug_s0/final CohereLabs/tiny-aya-global > logs/p4_merge_Dwaug_s0.log 2>&1 || exit 1
for sp in s4_api s4_wf4 p1_test_full p1_test_pp p1_test_rec p1_test_wf4 p1_test_combo; do
  [ -f results/p4/Dwaug_s0__$sp.summary.json ] && { echo "skip $sp"; continue; }
  .venv/bin/python scripts/p4_eval.py --model runs/p4_dwaug_s0/merged --tag Dwaug_s0 --split $sp --samples 4 > logs/p4_Dwaug_s0_$sp.log 2>&1 || { tail -30 logs/p4_Dwaug_s0_$sp.log; exit 1; }
  tail -1 logs/p4_Dwaug_s0_$sp.log
done
echo P4_Dwaug_s0_OK
