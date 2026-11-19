#!/bin/bash
# Launch one queue task on HF Jobs (run on the cluster login node). Claims queue/claims/<task> first, so no Slurm
# worker runs the same task; releases the claim if the launch fails. Logs the spend line to docs/stage2/HF_CREDITS.md
# (in the local repo, by hand) -- this script prints it.
#   scripts/hf_launch.sh <task> <flavor> <timeout> "<outputs>" [input paths ...]
set -euo pipefail
cd /scratch/pabitra/actionproof
TASK=$1; FLAVOR=$2; TIMEOUT=$3; OUTPUTS=$4; shift 4
REPO=atharvgaur18/actionproof-jobs
export HF_HOME=/scratch/pabitra/.cache/huggingface HF_HUB_OFFLINE=0
test -f slurm/tasks/$TASK.sh
mkdir -p queue/claims .cache/hfjob
mkdir queue/claims/$TASK 2>/dev/null || { echo "already claimed: $(cat queue/claims/$TASK/owner 2>/dev/null)"; exit 1; }
trap 'rm -rf queue/claims/$TASK; echo "launch failed; claim released"' ERR
tar czf .cache/hfjob/code_$TASK.tar.gz actionproof scripts slurm data/bench/proc data/bench/c2 pyproject.toml \
  PREREGISTRATION_A12.md docs/stage2/VERIFIER_SPEC.md results/a12_hashes.json results/a12 tests/golden \
  $( [ -f PREREGISTRATION_A11.md ] && echo PREREGISTRATION_A11.md ) $( [ -f results/a11_hashes.json ] && echo results/a11_hashes.json )
.venv/bin/hf upload $REPO .cache/hfjob/code_$TASK.tar.gz bundle/code_$TASK.tar.gz --repo-type dataset --quiet
.venv/bin/hf upload $REPO slurm/hfjob_run.sh bundle/hfjob_run.sh --repo-type dataset --quiet
for p in "$@"; do .venv/bin/hf upload $REPO "$p" "inputs/$TASK/$p" --repo-type dataset --quiet; done
JOB=$(.venv/bin/hf jobs run --detach --flavor $FLAVOR --timeout $TIMEOUT --secrets HF_TOKEN --name ap-$TASK \
  -e AP_REPO=$REPO -e AP_TASK=$TASK -e "AP_OUTPUTS=$OUTPUTS" ghcr.io/astral-sh/uv:python3.12-bookworm \
  bash -c "uvx hf download $REPO bundle/hfjob_run.sh --repo-type dataset --local-dir /tmp/b && bash /tmp/b/bundle/hfjob_run.sh" | tail -1)
echo "hf:$JOB:$FLAVOR:$(date -Is)" > queue/claims/$TASK/owner
trap - ERR
echo "LAUNCHED $TASK job=$JOB flavor=$FLAVOR timeout=$TIMEOUT at $(date '+%Y-%m-%d %H:%M')"
