# minutes: 90
# D1 judge-fooling audit: transcript judges (our llm_judge prompt and AuditKIT TaskCompletion) with
# Aya Expanse 8B and 32B (FP8) on the stratified pool of claimed-"completed" episodes.
set -uo pipefail
source slurm/env.sh
[ -f results/judge/pool.jsonl ] || .venv/bin/python -m actionproof.eval.judge_audit build
for m in "aya8b:CohereLabs/aya-expanse-8b" "aya32b:CohereLabs/aya-expanse-32b"; do
  tag=${m%%:*}; model=${m#*:}
  [ -f results/judge/${tag}__summary.json ] || .venv/bin/python -m actionproof.eval.judge_audit judge --model $model --tag $tag \
    > logs/170_judge_$tag.full.log 2>&1
  python3 -c "
import json; d=json.load(open('results/judge/${tag}__summary.json'))
for j,t in d.items(): print(j, {k:v for k,v in t.items() if k.startswith('ALL') or k=='unparsed'})"
done
