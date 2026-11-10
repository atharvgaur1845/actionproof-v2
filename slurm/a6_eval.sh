#!/bin/bash
# Amendment 6 evaluation set for one model.   a6_eval.sh TAG BASE ADAPTER [PROMPT]
set -euo pipefail
TAG=$1; BASE=$2; AD=$3; P=${4:-neutral}; D=results/eval/a6; mkdir -p $D
for sp in e_unrecoverable e6_twin_rp e6_depth3 e_seen_refund e_seen_other e_heldout e_heldout_ctrl test_iid test_ood_fault; do
  case $sp in test_*) src="--split $sp"; mt=10; data="";; e6_*) data=data/bench/a6/$sp.jsonl; src="--tasks $data"; mt=18;;
    *) data=data/bench/a4/$sp.jsonl; src="--tasks $data"; mt=18;; esac
  [ "$P" != neutral ] && case $sp in e_unrecoverable|e6_twin_rp|e_heldout) ;; *) continue;; esac  # prompt baselines: 3 splits
  o=$D/${TAG}__${P}__$sp.jsonl
  cmd=".venv/bin/python -m actionproof.eval.run_eval_a4 --model $BASE --adapter $AD $src --prompt $P --max-turns $mt --out $o"
  [ -f ${o%.jsonl}.summary.json ] || $cmd > logs/a6_${TAG}_${P}_$sp.full.log 2>&1
  .venv/bin/python scripts/a4_log_run.py --id A6E_${TAG}_${P}_$sp --kind eval --cmd "$cmd" --adapter $AD ${data:+--data $data} --out $o > /dev/null
  python3 -c "import json;d=json.load(open('${o%.jsonl}.summary.json'));a=d['a4'];print('$TAG $P $sp','vsc',round(d['vsc'],3),'hfr',round(a.get('hfr',float('nan')),3),'pfr',round(a.get('pfr',float('nan')),3),'pwv_refund',a.get('pwv_issue_refund',{}).get('rate'),'calls',round(a['model_calls'],2))"
done
