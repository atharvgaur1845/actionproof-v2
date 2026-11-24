#!/bin/bash
# Sync this worktree's code to the cluster working copy (/scratch/pabitra/lab_theory). Results are pulled back by
# scripts/theory/pull.sh. Never touches /scratch/pabitra/actionproof.
set -euo pipefail
cd "$(dirname "$0")/../.."
git rev-parse HEAD > COMMIT
rsync -a --exclude .git --exclude .venv --exclude '/results/' --exclude '/data/' --exclude '/demo/' --exclude '/product/' \
  --exclude 'evaluation/llm_judge_committee/outputs_j2/' --exclude node_modules ./ sharanga:/scratch/pabitra/lab_theory/
ssh sharanga 'mkdir -p /scratch/pabitra/lab_theory/logs /scratch/pabitra/lab_theory/results/theory /scratch/pabitra/lab_theory/results/research'
