# minutes: 60
# after: 152_grpo_v2_outcome
# H4/H6/H9 transfer: trained agents (RFT v2 init, GRPO v2 pcstr_d / outcome / tool_trust, seed 0) on banking
# (test_ood_domain), Hindi, Hinglish and pressure variants. Neutral prompt, greedy. Exploratory (PREREG §2).
set -uo pipefail
source slurm/env.sh
mkdir -p results/eval/transfer
S="test_ood_domain test_iid_hi test_iid_hinglish test_iid_pressure test_iid_hinglish_pressure"
ev() {  # tag model adapter split
  local R=results/eval/transfer/$1__neutral__$4
  [ -f $R.summary.json ] || .venv/bin/python -m actionproof.eval.run_eval --model $2 --adapter $3 --split $4 \
    --prompt neutral --out $R.jsonl > logs/180_$1_$4.full.log 2>&1 || { echo "FAILED $1 $4"; tail -5 logs/180_$1_$4.full.log; return; }
  python3 -c "import json;d=json.load(open('$R.summary.json'));print('$1', '$4', {k:round(d[k],3) for k in ('vsc','fcr','ivr','report_rate','verify_calls')})"
}
for s in $S; do ev rft_tiny_v2 CohereLabs/tiny-aya-global runs/rft_tiny_v2/final $s; done  # also builds runs/rft_tiny_v2/merged
for a in pcstr_d outcome tool_trust; do
  for s in $S; do ev grpo_v2_$a runs/rft_tiny_v2/merged runs/grpo_tiny_v2_${a}_s0/final $s; done
done
