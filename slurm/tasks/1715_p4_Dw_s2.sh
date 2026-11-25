# minutes: 240
# P4 (PREREGISTRATION_P4.md): arm Dw_s2 — SFT (actionproof.train.rft, 2 epochs, seed 2) then P4 evaluation.
set -uo pipefail
source slurm/claim.sh || exit 0
source slurm/env.sh
.venv/bin/python scripts/freeze_p4.py --check || exit 1
mkdir -p logs results/p4
[ -f runs/p4_dw_s2/final/adapter_config.json ] || .venv/bin/python -m actionproof.train.rft --data data/rft/p1_dwarm_rows.jsonl --out runs/p4_dw_s2 --epochs 2 --seed 2 > logs/p4_Dw_s2.full.log 2>&1 || { tail -30 logs/p4_Dw_s2.full.log; exit 1; }
rm -rf runs/p4_dw_s2/checkpoint-*/optimizer.pt
[ -f runs/p4_dw_s2/merged/config.json ] || .venv/bin/python -m actionproof.train.merge runs/p4_dw_s2/final CohereLabs/tiny-aya-global > logs/p4_merge_Dw_s2.log 2>&1 || exit 1
for sp in s4_api s4_wf4 p1_test_full p1_test_pp p1_test_rec p1_test_wf4 p1_test_combo; do
  [ -f results/p4/Dw_s2__$sp.summary.json ] && { echo "skip $sp"; continue; }
  .venv/bin/python scripts/p4_eval.py --model runs/p4_dw_s2/merged --tag Dw_s2 --split $sp --samples 4 > logs/p4_Dw_s2_$sp.log 2>&1 || { tail -30 logs/p4_Dw_s2_$sp.log; exit 1; }
  tail -1 logs/p4_Dw_s2_$sp.log
done
echo P4_Dw_s2_OK
