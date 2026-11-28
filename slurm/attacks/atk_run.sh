# Attack suite ATK1 model run (docs/attacks/SPEC.md §8). Usage (from a task file):
#   bash slurm/attacks/atk_run.sh TAG SPLIT RULES MODEL [ADAPTER]
# Runs in AP_ATK_ROOT (default /scratch/pabitra/attacks: a copy of branch research/attack-suite), never in the frozen
# /scratch/pabitra/actionproof tree; models are read from there by absolute path. The eval script refuses
# attack_heldout without a frozen PREREGISTRATION_ATTACKS.md and never overwrites an existing output.
set -uo pipefail
TAG=$1 SPLIT=$2 RULES=$3 MODEL=$4 ADAPTER=${5:-}   # pass merged checkpoints; an adapter would be merged next to itself
cd "${AP_ATK_ROOT:-/scratch/pabitra/attacks}" || exit 1
PY=${AP_PY:-/scratch/pabitra/actionproof/.venv/bin/python}
mkdir -p logs results/attacks/models
if [ -f results/attacks/models/${TAG}__${SPLIT}.summary.json ]; then echo "done already: $TAG $SPLIT"; exit 0; fi
rm -f results/attacks/models/${TAG}__${SPLIT}.jsonl   # a partial file from a crashed run is never used
$PY scripts/attacks_eval.py --tag "$TAG" --split "$SPLIT" --rules "$RULES" --samples 2 --model "$MODEL" \
  ${ADAPTER:+--adapter "$ADAPTER"} > logs/atk_${TAG}__${SPLIT}.log 2>&1 || { tail -40 logs/atk_${TAG}__${SPLIT}.log; exit 1; }
tail -1 logs/atk_${TAG}__${SPLIT}.log
sha256sum results/attacks/models/${TAG}__${SPLIT}.jsonl
