# minutes: 300
# after: 332_a4_gate
# Amendment 4 X7 / H15 primary: middleware within the RFT v2 lineage. Every cell (incl. "none") is re-run with this
# runner so all cells share one config: dev splits at 10 turns, A4 splits at 18 turns. Greedy, neutral prompt.
set -euo pipefail
source slurm/env.sh
D=results/eval/a4/mw; mkdir -p $D
ev() {  # tag base adapter mode
  for sp in test_iid test_ood_fault e_heldout e_unrecoverable; do
    case $sp in test_*) src="--split $sp"; mt=10; data="";; *) src="--tasks data/bench/a4/$sp.jsonl"; mt=18; data=data/bench/a4/$sp.jsonl;; esac
    o=$D/$1__$4__$sp.jsonl
    cmd=".venv/bin/python -m actionproof.eval.run_eval_a4 --model $2 --adapter $3 $src --max-turns $mt --middleware $4 --out $o"
    [ -f ${o%.jsonl}.summary.json ] || $cmd > logs/350_$1_$4_$sp.full.log 2>&1
    .venv/bin/python scripts/a4_log_run.py --id X7_$1_$4_$sp --kind eval_mw --cmd "$cmd" --adapter $3 ${data:+--data $data} --out $o > /dev/null
    python3 -c "import json;d=json.load(open('${o%.jsonl}.summary.json'));a=d['a4'];print('$1 $4 $sp', 'vsc',round(d['vsc'],3),'fcr',round(d['fcr'],3),'dsr',round(d['dsr'],3),'cmr',round(a['cmr'],3),'calls',round(a['model_calls'],2),'+h',round(a['harness_calls'],2),'rs',round(a.get('rs',float('nan')),3),'hfr',round(a.get('hfr',float('nan')),3))"
  done
}
for m in none raw vbr raw+vbr; do
  ev init CohereLabs/tiny-aya-global runs/rft_tiny_v2/final $m
  ev outcome_s0 runs/rft_tiny_v2/merged runs/grpo_tiny_v2_outcome_s0/final $m
  ev pcstr_d_s0 runs/rft_tiny_v2/merged runs/grpo_tiny_v2_pcstr_d_s0/final $m
done
for m in none raw; do ev tool_trust_s0 runs/rft_tiny_v2/merged runs/grpo_tiny_v2_tool_trust_s0/final $m; done
