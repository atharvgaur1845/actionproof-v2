# minutes: 90
# Task 416b (exploratory, SPEC §27 2026-11-13): (1) refund-UR rollouts in the TRAINING schema (plain EcomEnv; D-EVAL-1
# measured, not inferred from the battery); (2) A7 warm start on test_iid and e_seen_refund (the "before" for §5c).
# Dual-pool copy (h200 + a100): run once only. If the other pool already claimed or finished this task, skip.
for f in /scratch/pabitra/actionproof/queue/*/running/$(basename "$0") /scratch/pabitra/actionproof/queue/*/done/$(basename "$0"); do
  [ -e "$f" ] && [ "$(readlink -f "$f")" != "$(readlink -f "$0")" ] && { echo "SKIP: $(basename "$0") already claimed or done at $f"; exit 0; }
done
set -euo pipefail
source slurm/env.sh
D=results/eval/a7/plain; mkdir -p $D; A6=runs/rft_tiny_a6/merged; UR=data/bench/a4/e_unrecoverable.jsonl
pe() {  # tag adapter-or-empty
  o=$D/${1}__neutral__e_unrecoverable.jsonl
  [ -f ${o%.jsonl}.summary.json ] || .venv/bin/python scripts/eval_plain_schema.py --model $A6 ${2:+--adapter $2} \
    --tasks $UR --max-turns 18 --out $o > logs/416b_plain_$1.full.log 2>&1
  python3 -c "import json;d=json.load(open('${o%.jsonl}.summary.json'));a=d['a4'];print('plain $1 n',d['n'],'vsc',round(d['vsc'],3),'hfr',a.get('hfr'))"
}
pe init ""
for a in pcstr_d_s0 pcstr_d_noreport_s0 ack_outcome_s0 pcstr_d_s1 pcstr_d_noreport_s1 pcstr_d_noND_s0 pcstr_d_noFunder_s0 outcome_s0; do
  pe a6_$a runs/grpo_tiny_a6_$a/final; done
E=results/eval/a7
for sp in test_iid e_seen_refund; do
  case $sp in test_*) src="--split $sp"; mt=10;; *) src="--tasks data/bench/a4/$sp.jsonl"; mt=18;; esac
  o=$E/a7_init__neutral__$sp.jsonl
  [ -f ${o%.jsonl}.summary.json ] || .venv/bin/python -m actionproof.eval.run_eval_a4 --model CohereLabs/tiny-aya-global \
    --adapter runs/rft_tiny_a7/final $src --max-turns $mt --out $o > logs/416b_a7_init_$sp.full.log 2>&1
  python3 -c "import json;d=json.load(open('${o%.jsonl}.summary.json'));a=d['a4'];print('a7_init $sp vsc',round(d['vsc'],3),'pwv_refund',a.get('pwv_issue_refund',{}).get('rate'))"
done
