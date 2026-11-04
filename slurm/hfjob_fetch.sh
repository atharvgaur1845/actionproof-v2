#!/bin/bash
# Pull HF Job outputs (see hfjob_rft_v2.sh) into the cluster tree and mark the queue tasks done,
# which releases the GRPO arms (their `# after: 091_curriculum_v2` dependency) to cluster workers.
set -euo pipefail
cd /scratch/pabitra/actionproof
export HF_HOME=/scratch/pabitra/.cache/huggingface
.venv/bin/hf download atharvgaur18/actionproof-jobs --repo-type dataset --include "outputs/*" --local-dir .cache/hfjob/dl --quiet
rsync -a .cache/hfjob/dl/outputs/ ./
test -f runs/rft_tiny_v2/final/adapter_config.json && test -s data/curriculum/band_tiny_v2.jsonl
for t in 087_rft_v2_tiny 091_curriculum_v2; do
  echo "completed on HF Jobs ($(date -Is)); outputs fetched by slurm/hfjob_fetch.sh" > queue/h200/logs/$t.log
  mv queue/h200/hfjobs/$t.sh queue/h200/done/
done
wc -l data/curriculum/band_tiny_v2.jsonl
