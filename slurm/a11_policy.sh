#!/bin/bash
# Amendment 11 (PREREGISTRATION_A11, H1): one policy through the e-commerce gateway on both c2 splits, in the three
# live modes. Observe is computed offline from the passthrough episodes (task 509), never re-generated.
#   slurm/a11_policy.sh TAG MODEL [ADAPTER] [EXTRA_ARGS]
#   e.g. slurm/a11_policy.sh pcstr_d_s0 runs/rft_tiny_v2/merged runs/grpo_tiny_v2_pcstr_d_s0/final
set -uo pipefail
TAG=$1; MODEL=$2; AD=${3:-}; EXTRA=${4:-}
mkdir -p results/a11/eval logs
# On HF Jobs the merged Stage-1 warm start is rebuilt from its adapter (same merge as on the cluster).
if [ "$MODEL" = runs/rft_tiny_v2/merged ] && [ ! -f runs/rft_tiny_v2/merged/config.json ]; then
  .venv/bin/python -m actionproof.train.merge runs/rft_tiny_v2/final CohereLabs/tiny-aya-global > logs/a11_merge.log 2>&1 || exit 1
fi
for sp in test_iid_c2 v2_stress_c2; do
  for mode in passthrough enforce enforce_inform; do
    o=results/a11/eval/${TAG}__${mode}__${sp}.jsonl
    [ -f ${o%.jsonl}.summary.json ] || .venv/bin/python scripts/eval_gateway.py --model $MODEL ${AD:+--adapter $AD} \
      --mode $mode --split $sp --prompt neutral --max-turns 12 --seed 0 $EXTRA --out $o \
      > logs/a11_${TAG}_${mode}_${sp}.full.log 2>&1
    python3 -c "import json;d=json.load(open('${o%.jsonl}.summary.json'));print('$TAG $mode $sp', 'system_vsc', round(d['system']['vsc'],3), 'agent_vsc', round(d['agent']['vsc'],3), 'false_final', round(d['system']['false_finalization'],3))" \
      || { echo "$TAG $mode $sp FAILED (see logs/a11_${TAG}_${mode}_${sp}.full.log)"; exit 1; }
  done
done
