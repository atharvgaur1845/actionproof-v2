# minutes: 120
# Amendment 9 (PREREGISTRATION_A9 §2): judge audit with the receipt ledger; 32B transcript-only re-run for the paired split.
set -uo pipefail
source slurm/env.sh
.venv/bin/python scripts/freeze_a9.py --check || exit 1
for r in "aya8b_grounded:CohereLabs/aya-expanse-8b:--grounded" "aya32b_t:CohereLabs/aya-expanse-32b:" "aya32b_grounded:CohereLabs/aya-expanse-32b:--grounded"; do
  IFS=: read tag model flag <<< "$r"
  [ -f results/judge/${tag}__summary.json ] || .venv/bin/python -m actionproof.eval.judge_audit judge --model $model --tag $tag $flag \
    > logs/420_judge_$tag.full.log 2>&1
  python3 -c "
import json; d=json.load(open('results/judge/${tag}__summary.json'))
for j,t in d.items(): print(j, {k:v for k,v in t.items() if k.startswith('ALL') or k=='unparsed'})"
done
