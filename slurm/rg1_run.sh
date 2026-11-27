# RG1 sealed-split run (PREREGISTRATION_RG1.md). Usage (from a queue task):
#   bash slurm/rg1_run.sh TAG SPLIT RULES MODEL [ADAPTER]
# The eval script runs scripts/freeze_rg1.py --check first and refuses to overwrite an existing output.
set -uo pipefail
TAG=$1 SPLIT=$2 RULES=$3 MODEL=$4 ADAPTER=${5:-}
mkdir -p logs results/rg1
if [ -f results/rg1/${TAG}__${SPLIT}.summary.json ]; then echo "done already: $TAG $SPLIT"; exit 0; fi
rm -f results/rg1/${TAG}__${SPLIT}.jsonl   # a partial file from a crashed run is never used
.venv/bin/python scripts/rg1_eval.py --tag "$TAG" --split "$SPLIT" --rules "$RULES" --samples 2 --model "$MODEL" \
  ${ADAPTER:+--adapter "$ADAPTER"} > logs/rg1_${TAG}__${SPLIT}.log 2>&1 || { tail -40 logs/rg1_${TAG}__${SPLIT}.log; exit 1; }
tail -1 logs/rg1_${TAG}__${SPLIT}.log
sha256sum results/rg1/${TAG}__${SPLIT}.jsonl
