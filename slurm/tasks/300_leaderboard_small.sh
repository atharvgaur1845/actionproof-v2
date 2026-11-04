# minutes: 180
# R0 leaderboard, models <= 8B (H100 pool): 6 models x {neutral, verify_a} on test_iid; Tiny Global and
# Aya 8B also on test_ood_fault, test_ood_domain (banking), test_iid_hi / _hinglish / _pressure (neutral).
set -uo pipefail
source slurm/env.sh
run() {  # model prompt split [gpu-mem]
  local m=$1 p=$2 s=$3 g=${4:-0.85} o=results/eval/lb/$(basename $1)__$2__$3.jsonl
  [ -f "${o%.jsonl}.summary.json" ] && { echo "skip $o"; return; }
  .venv/bin/python -m actionproof.eval.run_eval --model $m --prompt $p --split $s --gpu-mem $g --out $o \
    > logs/lb_$(basename $m)_${p}_${s}.full.log 2>&1 && echo "done $o $(python3 -c "import json;d=json.load(open('${o%.jsonl}.summary.json'));print('VSC',round(d['vsc'],3),'FCR',round(d['fcr'],3))")" || echo "FAILED $o"
}
for m in CohereLabs/tiny-aya-global CohereLabs/aya-expanse-8b CohereLabs/tiny-aya-fire CohereLabs/tiny-aya-water \
         CohereLabs/tiny-aya-earth CohereLabs/tiny-aya-base; do
  for p in neutral verify_a; do run $m $p test_iid; done
done
for m in CohereLabs/tiny-aya-global CohereLabs/aya-expanse-8b; do
  for p in neutral verify_a; do run $m $p test_ood_fault; done
  for s in test_ood_domain test_iid_hi test_iid_hinglish test_iid_pressure test_iid_hinglish_pressure; do run $m neutral $s; done
done
