# minutes: 60
# H5: probe-triggered verification guard on the RFT init (no retraining) vs neutral and always-verify,
# on test_iid. Probe = pos-A probe trained on RFT train-distribution samples; tau fixed on val (80% recall).
set -euo pipefail
source slurm/env.sh
M="--model CohereLabs/tiny-aya-global --adapter runs/rft_tiny_curated/final --split test_iid"
for cfg in "neutral:" "verify_a:" "neutral:--guard-probe results/interp/probe_A.npz"; do
  p=${cfg%%:*}; g=${cfg#*:}; tag=$p$([ -n "$g" ] && echo _guard)
  o=results/eval/r2/rft_tiny__${tag}__test_iid.jsonl
  [ -f ${o%.jsonl}.summary.json ] || .venv/bin/python -m actionproof.eval.run_eval $M --prompt $p $g --out $o > logs/130_$tag.full.log 2>&1
  python3 -c "import json;d=json.load(open('${o%.jsonl}.summary.json'));print('$tag', {k:(round(d[k],3) if isinstance(d[k],float) else d[k]) for k in ('vsc','fcr','ivr','dsr','verify_calls','nudges_per_ep','guard')})"
done
