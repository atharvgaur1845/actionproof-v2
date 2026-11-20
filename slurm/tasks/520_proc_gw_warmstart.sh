# minutes: 240
# Stage 2 (exploratory, PLAN H1 on ProcureSim): a12_init through the gateway in all four modes on proc_dev (never a confirmatory split).
set -uo pipefail
source slurm/claim.sh || exit 0
source slurm/env.sh
mkdir -p results/eval/proc_gateway logs
for mode in passthrough observe enforce enforce_inform; do
  o=results/eval/proc_gateway/a12_init__${mode}__proc_dev.jsonl
  [ -f ${o%.jsonl}.summary.json ] || .venv/bin/python scripts/eval_proc_gateway.py --model CohereLabs/tiny-aya-global --adapter runs/rft_proc/final --split proc_dev --mode $mode --out $o > logs/520_proc_gw_warmstart_${mode}.full.log 2>&1 || { tail -20 logs/520_proc_gw_warmstart_${mode}.full.log; exit 1; }
done
