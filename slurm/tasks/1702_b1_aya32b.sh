# minutes: 300
# B1 (PREREGISTRATION_B1.md): prompted CohereLabs/aya-expanse-32b on the five frozen P1 test splits; results/b1/.
set -uo pipefail
source slurm/claim.sh || exit 0
source slurm/env.sh
.venv/bin/python scripts/freeze_b1.py --check || exit 1
mkdir -p logs results/b1
for sp in p1_test_full p1_test_pp p1_test_rec p1_test_wf4 p1_test_combo; do
  [ -f results/b1/aya32b__$sp.summary.json ] && { echo "skip $sp"; continue; }
  rm -f results/b1/aya32b__$sp.jsonl
  .venv/bin/python scripts/b1_eval.py --model CohereLabs/aya-expanse-32b --tag aya32b --split $sp --samples 4 --fp8 > logs/b1_aya32b_$sp.log 2>&1 || { tail -30 logs/b1_aya32b_$sp.log; exit 1; }
  tail -1 logs/b1_aya32b_$sp.log
done
echo B1_aya32b_OK
