# minutes: 240
# Stage 2 (exploratory, PLAN H1 on ProcureSim): aya32b through the gateway in all four modes on proc_dev (never a confirmatory split).
set -uo pipefail
source slurm/claim.sh || exit 0
source slurm/env.sh
mkdir -p results/eval/proc_gateway logs
for mode in passthrough observe enforce enforce_inform; do
  o=results/eval/proc_gateway/aya32b__${mode}__proc_dev.jsonl
  [ -f ${o%.jsonl}.summary.json ] || .venv/bin/python scripts/eval_proc_gateway.py --model CohereLabs/aya-expanse-32b  --split proc_dev --mode $mode --out $o > logs/522_proc_gw_aya32b_${mode}.full.log 2>&1 || { tail -20 logs/522_proc_gw_aya32b_${mode}.full.log; exit 1; }
done
