# minutes: 240
# R0 leaderboard, Aya Expanse 32B (H200 pool): test_iid + test_ood_fault x {neutral, verify_a}, and the new
# axes (neutral): banking, HI, Hinglish, pressure.
set -uo pipefail
source slurm/env.sh
run() {  # model prompt split [gpu-mem]
  local m=$1 p=$2 s=$3 g=${4:-0.85} o=results/eval/lb/$(basename $1)__$2__$3.jsonl
  [ -f "${o%.jsonl}.summary.json" ] && { echo "skip $o"; return; }
  .venv/bin/python -m actionproof.eval.run_eval --model $m --prompt $p --split $s --gpu-mem $g --out $o \
    > logs/lb_$(basename $m)_${p}_${s}.full.log 2>&1 && echo "done $o $(python3 -c "import json;d=json.load(open('${o%.jsonl}.summary.json'));print('VSC',round(d['vsc'],3),'FCR',round(d['fcr'],3))")" || echo "FAILED $o"
}
m=CohereLabs/aya-expanse-32b
for s in test_iid test_ood_fault; do for p in neutral verify_a; do run $m $p $s 0.88; done; done
for s in test_ood_domain test_iid_hi test_iid_hinglish test_iid_pressure test_iid_hinglish_pressure; do run $m neutral $s 0.88; done
