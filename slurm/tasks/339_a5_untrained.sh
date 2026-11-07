# minutes: 20
# after: 333_gates
# A5 baseline: untrained Tiny Aya Global (neutral), a5_eval.
set -euo pipefail
source slurm/env.sh
R=results/eval/a5/tiny_untrained__neutral__none__a5_eval
[ -f $R.summary.json ] || .venv/bin/python -m actionproof.eval.run_eval_a5 --model CohereLabs/tiny-aya-global --tasks data/bench/a5/a5_eval.jsonl --max-turns 18 --out $R.jsonl > logs/365.full.log 2>&1
python3 -c "import json;print(json.load(open('$R.summary.json'))['a5']['all'])"
