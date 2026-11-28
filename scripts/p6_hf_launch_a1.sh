#!/bin/bash
# Addendum 1 variant of scripts/p6_hf_launch.sh (which stays frozen): H200 with a 2h30m timeout, seeds 0-1 only.
# Each job runs 2300_p6_<arm>_s<k> then 2310_p6_eval_<arm>_s<k>; outputs land in outputs/p6_<arm>_s<k>/ of $REPO.
#   scripts/p6_hf_launch_a1.sh   (addendum 1: seeds 0-1 of both arms on HF; seed 2 runs on the cluster)
set -euo pipefail
cd "$(dirname "$0")/.."
REPO=atharvgaur18/actionproof-jobs
FLAVOR=h200; TIMEOUT=2h30m   # addendum 1: four jobs, at most $50
VENV=/scratch/pabitra/actionproof/.venv   # read-only use of the main environment's Python and hf CLI
export HF_HOME=/scratch/pabitra/.cache/huggingface HF_HUB_OFFLINE=0
$VENV/bin/python scripts/freeze_p6.py --check
mkdir -p .cache/p6
tar czf .cache/p6/code_p6.tar.gz actionproof scripts slurm data/bench pyproject.toml PREREGISTRATION_*.md \
  docs/stage2/VERIFIER_SPEC.md docs/stage2/OPSIM_SPEC.md results/*_hashes.json results/p4/Dwaug_s*__*.jsonl \
  runs/p4_dwaug_s0/final runs/p4_dwaug_s1/final runs/p4_dwaug_s2/final tests/test_p6_stats.py
$VENV/bin/hf upload $REPO .cache/p6/code_p6.tar.gz bundle/code_p6.tar.gz --repo-type dataset --quiet
$VENV/bin/hf upload $REPO slurm/p6_hfjob_run.sh bundle/p6_hfjob_run.sh --repo-type dataset --quiet
for job in ${@:-can_s0 can_s1 surf_s0 surf_s1}; do
  arm=${job%_s*}; seed=${job#*_s}; full=$([ "$arm" = can ] && echo canonical || echo surface)
  ID=$($VENV/bin/hf jobs run --detach --flavor $FLAVOR --timeout $TIMEOUT --secrets HF_TOKEN --name ap-p6-$job \
    -e AP_REPO=$REPO -e AP_TAG=p6_$job -e "AP_TASKS=2300_p6_${job},2310_p6_eval_${job}" \
    -e "AP_OUTPUTS=runs/p6_${full}_s${seed}/final results/p6" ghcr.io/astral-sh/uv:python3.12-bookworm \
    bash -c "uvx hf download $REPO bundle/p6_hfjob_run.sh --repo-type dataset --local-dir /tmp/b && bash /tmp/b/bundle/p6_hfjob_run.sh" | tail -1)
  echo "LAUNCHED p6_$job job=$ID flavor=$FLAVOR timeout=$TIMEOUT at $(date '+%Y-%m-%d %H:%M %Z')"
done
