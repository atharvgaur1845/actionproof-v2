# minutes: 120
# Task 416a (exploratory, SPEC §27 2026-11-13): corrected forced-prefix battery on the evaluation backend.
# Registered battery scored a lone '"' after each candidate; real outputs emit '",' (one token). Rescore with '",'
# via vLLM prompt logprobs on the merged weights, plus the greedy next action on every prefix (validity check).
# Dual-pool copy (h200 + a100): run once only. If the other pool already claimed or finished this task, skip.
for f in /scratch/pabitra/actionproof/queue/*/running/$(basename "$0") /scratch/pabitra/actionproof/queue/*/done/$(basename "$0"); do
  [ -e "$f" ] && [ "$(readlink -f "$f")" != "$(readlink -f "$0")" ] && { echo "SKIP: $(basename "$0") already claimed or done at $f"; exit 0; }
done
set -euo pipefail
source slurm/env.sh
P=data/bench/a7/forced_prefixes.jsonl; G=results/a7/battery_v2_greedy.jsonl; S=results/a7/battery_v2_scores.jsonl
has() { [ -f $2 ] && python3 -c "import json,sys;sys.exit(0 if any(json.loads(l)['model']=='$1' for l in open('$2')) else 1)"; }
run() {  # tag weights base
  [ -f $2/config.json ] || .venv/bin/python -c "from actionproof.train.merge import merge; merge('${2%/merged}/final','$3')"
  has $1 $G || .venv/bin/python scripts/battery_vllm.py --mode greedy --prefixes $P --tag $1 --weights $2 --out $G >> logs/416a_battery_v2.full.log 2>&1
  has $1 $S || .venv/bin/python scripts/battery_vllm.py --mode score --prefixes $P --tag $1 --weights $2 --out $S >> logs/416a_battery_v2.full.log 2>&1
  echo "$(date +%T) done $1"
}
A6=runs/rft_tiny_a6/merged; A7=runs/rft_tiny_a7/merged
run init $A6
run a6_pcstr_d_noreport_s0 runs/grpo_tiny_a6_pcstr_d_noreport_s0/merged $A6
run a6_pcstr_d_s0 runs/grpo_tiny_a6_pcstr_d_s0/merged $A6
run a6_ack_outcome_s0 runs/grpo_tiny_a6_ack_outcome_s0/merged $A6
run a7_init $A7
run a7_pcstr_d_noreport_s0 runs/grpo_tiny_a7_pcstr_d_noreport_s0/merged $A7
run a7_ack_outcome_s0 runs/grpo_tiny_a7_ack_outcome_s0/merged $A7
for a in pcstr_d_s1 pcstr_d_noreport_s1 pcstr_d_noND_s0 pcstr_d_noFunder_s0 outcome_s0; do
  run a6_$a runs/grpo_tiny_a6_$a/merged $A6; done
.venv/bin/python scripts/battery_v2_report.py $S $G results/a7/forced_prefix_all.jsonl > results/a7/battery_v2.md || echo "report failed (run locally)"
