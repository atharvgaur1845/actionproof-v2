# minutes: 240
# P5 (PREREGISTRATION_P5.md): recovery SFT on CohereLabs/tiny-aya-water; base and Dw_water on the frozen P1 test.
set -uo pipefail
source slurm/claim.sh || exit 0
source slurm/env.sh
.venv/bin/python scripts/freeze_p5.py --check || exit 1
sha256sum -c data/rft/p1_dwarm_rows.jsonl.sha256 || exit 1
mkdir -p logs results/p5
SPLITS="p1_test_full p1_test_pp p1_test_rec p1_test_wf4 p1_test_combo"
for sp in $SPLITS; do [ -f results/p5/base_water__$sp.summary.json ] || .venv/bin/python scripts/p5_eval.py --model CohereLabs/tiny-aya-water --tag base_water --split $sp --samples 4 > logs/p5_base_water_$sp.log 2>&1 || { tail -20 logs/p5_base_water_$sp.log; exit 1; }; done
[ -f runs/p5_dw_water/final/adapter_config.json ] || .venv/bin/python -m actionproof.train.rft --data data/rft/p1_dwarm_rows.jsonl --model CohereLabs/tiny-aya-water --out runs/p5_dw_water --epochs 2 --seed 0 > logs/p5_dw_water.full.log 2>&1 || { tail -30 logs/p5_dw_water.full.log; exit 1; }
rm -rf runs/p5_dw_water/checkpoint-*/optimizer.pt
[ -f runs/p5_dw_water/merged/config.json ] || .venv/bin/python -m actionproof.train.merge runs/p5_dw_water/final CohereLabs/tiny-aya-water > logs/p5_merge_water.log 2>&1 || exit 1
for sp in $SPLITS; do [ -f results/p5/dw_water__$sp.summary.json ] || .venv/bin/python scripts/p5_eval.py --model runs/p5_dw_water/merged --tag dw_water --split $sp --samples 4 > logs/p5_dw_water_$sp.log 2>&1 || { tail -20 logs/p5_dw_water_$sp.log; exit 1; }; done
echo P5_water_OK
