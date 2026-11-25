# minutes: 240
# B1 deviation D-B1-1: Aya Expanse 32B rerun in bf16 (the fp8 run produced degenerate output).
set -uo pipefail
source slurm/claim.sh || exit 0
source slurm/env.sh
.venv/bin/python scripts/freeze_b1.py --check || exit 1
for sp in p1_test_full p1_test_pp p1_test_rec p1_test_wf4 p1_test_combo; do
  [ -f results/b1/aya32b_bf16__$sp.summary.json ] && continue
  rm -f results/b1/aya32b_bf16__$sp.jsonl
  .venv/bin/python scripts/b1_eval.py --model CohereLabs/aya-expanse-32b --tag aya32b_bf16 --split $sp --samples 4 --gpu-mem 0.95 > logs/b1_aya32b_bf16_$sp.log 2>&1 || { tail -30 logs/b1_aya32b_bf16_$sp.log; exit 1; }
  tail -1 logs/b1_aya32b_bf16_$sp.log
done
echo B1_aya32b_bf16_OK
