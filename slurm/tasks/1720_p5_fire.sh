# minutes: 240
# P5 (PREREGISTRATION_P5.md): recovery SFT on CohereLabs/tiny-aya-fire; base and Dw_fire on the frozen P1 test.
set -uo pipefail
source slurm/claim.sh || exit 0
source slurm/env.sh
.venv/bin/python scripts/freeze_p5.py --check || exit 1
sha256sum -c data/rft/p1_dwarm_rows.jsonl.sha256 || exit 1
mkdir -p logs results/p5
SPLITS="p1_test_full p1_test_pp p1_test_rec p1_test_wf4 p1_test_combo"
for sp in $SPLITS; do [ -f results/p5/base_fire__$sp.summary.json ] || .venv/bin/python scripts/p5_eval.py --model CohereLabs/tiny-aya-fire --tag base_fire --split $sp --samples 4 > logs/p5_base_fire_$sp.log 2>&1 || { tail -20 logs/p5_base_fire_$sp.log; exit 1; }; done
[ -f runs/p5_dw_fire/final/adapter_config.json ] || .venv/bin/python -m actionproof.train.rft --data data/rft/p1_dwarm_rows.jsonl --model CohereLabs/tiny-aya-fire --out runs/p5_dw_fire --epochs 2 --seed 0 > logs/p5_dw_fire.full.log 2>&1 || { tail -30 logs/p5_dw_fire.full.log; exit 1; }
rm -rf runs/p5_dw_fire/checkpoint-*/optimizer.pt
[ -f runs/p5_dw_fire/merged/config.json ] || .venv/bin/python -m actionproof.train.merge runs/p5_dw_fire/final CohereLabs/tiny-aya-fire > logs/p5_merge_fire.log 2>&1 || exit 1
for sp in $SPLITS; do [ -f results/p5/dw_fire__$sp.summary.json ] || .venv/bin/python scripts/p5_eval.py --model runs/p5_dw_fire/merged --tag dw_fire --split $sp --samples 4 > logs/p5_dw_fire_$sp.log 2>&1 || { tail -20 logs/p5_dw_fire_$sp.log; exit 1; }; done
echo P5_fire_OK
