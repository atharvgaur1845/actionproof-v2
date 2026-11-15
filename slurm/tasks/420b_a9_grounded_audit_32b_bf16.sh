# minutes: 90
# Amendment 9 deviation D-A9-1: task 420's 32B runs used FP8 on an A100 and produced garbage. Same audit in bf16 on A100
# (FP8 kept on H100/H200). Only actionproof/eval/judge_audit.py changed (the quantization line), so the A9 freeze check
# reports that one file; it is logged, not re-registered. Dual-pool copy: run once.
for f in /scratch/pabitra/actionproof/queue/*/running/420b_*.sh /scratch/pabitra/actionproof/queue/*/done/420b_*.sh; do
  [ -e "$f" ] && [ "$(readlink -f "$f")" != "$(readlink -f "$0")" ] && { echo "SKIP: $f"; exit 0; }
done
set -uo pipefail
source slurm/env.sh
.venv/bin/python scripts/freeze_a9.py --check
for tag in aya32b_t aya32b_grounded; do
  flag=""; [ $tag = aya32b_grounded ] && flag=--grounded
  [ -f results/judge/${tag}__summary.json ] || .venv/bin/python -m actionproof.eval.judge_audit judge --model CohereLabs/aya-expanse-32b \
    --tag $tag $flag > logs/420b_judge_$tag.full.log 2>&1
  python3 -c "
import json; d=json.load(open('results/judge/${tag}__summary.json'))
for j,t in d.items(): print(j, {k:v for k,v in t.items() if k.startswith('ALL') or k=='unparsed'})"
done
