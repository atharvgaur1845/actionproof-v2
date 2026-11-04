# minutes: 60
# after: 087_rft_v2_tiny
# H1 baseline-selection inputs (PREREGISTRATION §2): val VSC for base and RFT v2 under each verify
# prompt (+ RFT v2 neutral comes from 087). Selection itself is done by scripts, on val only.
set -uo pipefail
source slurm/env.sh
for m in "base_tiny:--model CohereLabs/tiny-aya-global" "rft_tiny_v2:--model CohereLabs/tiny-aya-global --adapter runs/rft_tiny_v2/final"; do
  tag=${m%%:*}; args=${m#*:}
  for p in verify_a verify_b verify_c; do
    R=results/eval/r1/${tag}__${p}__val
    [ -f $R.summary.json ] || .venv/bin/python -m actionproof.eval.run_eval $args --split val --prompt $p --out $R.jsonl > logs/160_${tag}_$p.full.log 2>&1
    python3 -c "import json;d=json.load(open('$R.summary.json'));print('$tag $p', {k:round(d[k],3) for k in ('vsc','fcr','report_rate','verify_calls')})"
  done
done
