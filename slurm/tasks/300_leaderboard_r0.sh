# minutes: 150
# R0 leaderboard: all 7 permitted Cohere models x {neutral, verify_a} on test_iid (200 tasks),
# plus test_ood_fault for Tiny Global, Aya 8B, Aya 32B. Greedy. Resumable: skips finished cells.
set -uo pipefail
source slurm/env.sh
run() {  # model prompt split [gpu-mem]
  local m=$1 p=$2 s=$3 g=${4:-0.85} o=results/eval/lb/$(basename $1)__$2__$3.jsonl
  [ -f "${o%.jsonl}.summary.json" ] && { echo "skip $o"; return; }
  .venv/bin/python -m actionproof.eval.run_eval --model $m --prompt $p --split $s --gpu-mem $g --out $o \
    > logs/lb_$(basename $m)_$p_$s.full.log 2>&1 && echo "done $o $(python3 -c "import json;d=json.load(open('${o%.jsonl}.summary.json'));print('VSC',round(d['vsc'],3),'FCR',round(d['fcr'],3))")" || echo "FAILED $o"
}
for m in CohereLabs/tiny-aya-global CohereLabs/aya-expanse-8b CohereLabs/aya-expanse-32b \
         CohereLabs/tiny-aya-fire CohereLabs/tiny-aya-water CohereLabs/tiny-aya-earth CohereLabs/tiny-aya-base; do
  g=0.85; [[ $m == *32b ]] && g=0.88
  for p in neutral verify_a; do run $m $p test_iid $g; done
done
for m in CohereLabs/tiny-aya-global CohereLabs/aya-expanse-8b CohereLabs/aya-expanse-32b; do
  g=0.85; [[ $m == *32b ]] && g=0.88
  for p in neutral verify_a; do run $m $p test_ood_fault $g; done
done
.venv/bin/python -m actionproof.eval.leaderboard --glob 'results/eval/lb/*.jsonl' --out results/leaderboard
