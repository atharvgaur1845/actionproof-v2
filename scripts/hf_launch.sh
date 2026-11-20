#!/bin/bash
# Launch one queue task on HF Jobs (run on the cluster login node). Claims queue/claims/<task> first, so no Slurm
# worker runs the same task; releases the claim if the launch fails. Logs the spend line to docs/stage2/HF_CREDITS.md
# (in the local repo, by hand) -- this script prints it.
#   scripts/hf_launch.sh <task> <flavor> <timeout> "<outputs>" [input paths ...]
set -euo pipefail
cd /scratch/pabitra/actionproof
TASKS=$1; FLAVOR=$2; TIMEOUT=$3; OUTPUTS=$4; shift 4   # TASKS: one task or a comma-separated list run in order
TASK=${TASKS%%,*}
REPO=atharvgaur18/actionproof-jobs
export HF_HOME=/scratch/pabitra/.cache/huggingface HF_HUB_OFFLINE=0
mkdir -p queue/claims .cache/hfjob
CLAIMED=""
release() { for t in $CLAIMED; do rm -rf queue/claims/$t; done; echo "launch failed; claims released"; }
for t in ${TASKS//,/ }; do
  test -f slurm/tasks/$t.sh
  if mkdir queue/claims/$t 2>/dev/null; then CLAIMED="$CLAIMED $t"; else echo "already claimed: $t $(cat queue/claims/$t/owner 2>/dev/null)"; release; exit 1; fi
done
trap release ERR
# every registration, hash manifest and test a task's guard may check travels with the code (701 failed without A14's)
tar czf .cache/hfjob/code_$TASK.tar.gz --exclude=results/a12/rollouts actionproof scripts slurm data/bench/proc data/bench/c2 pyproject.toml \
  PREREGISTRATION_*.md docs/stage2/VERIFIER_SPEC.md results/*_hashes.json results/a12 \
  $( [ -d data/bench/opsim ] && echo data/bench/opsim ) $( [ -d results/k1 ] && echo results/k1 ) $( [ -d results/a12b ] && echo results/a12b ) \
  $( [ -d results/a14 ] && echo results/a14 ) $( [ -d results/s2 ] && echo results/s2 ) tests/golden tests/integrity \
  $( [ -f PREREGISTRATION_A11.md ] && echo data/h4 )
.venv/bin/hf upload $REPO .cache/hfjob/code_$TASK.tar.gz bundle/code_$TASK.tar.gz --repo-type dataset --quiet
.venv/bin/hf upload $REPO slurm/hfjob_run.sh bundle/hfjob_run.sh --repo-type dataset --quiet
for p in "$@"; do .venv/bin/hf upload $REPO "$p" "inputs/$TASK/$p" --repo-type dataset --quiet; done
JOB=$(.venv/bin/hf jobs run --detach --flavor $FLAVOR --timeout $TIMEOUT --secrets HF_TOKEN --name ap-$TASK \
  -e AP_REPO=$REPO -e AP_TASK=$TASK -e "AP_TASKS=$TASKS" -e "AP_OUTPUTS=$OUTPUTS" ghcr.io/astral-sh/uv:python3.12-bookworm \
  bash -c "uvx hf download $REPO bundle/hfjob_run.sh --repo-type dataset --local-dir /tmp/b && bash /tmp/b/bundle/hfjob_run.sh" | tail -1)
for t in $CLAIMED; do echo "hf:$JOB:$FLAVOR:$(date -Is)" > queue/claims/$t/owner; done
trap - ERR
echo "LAUNCHED $TASKS job=$JOB flavor=$FLAVOR timeout=$TIMEOUT at $(date '+%Y-%m-%d %H:%M')"
