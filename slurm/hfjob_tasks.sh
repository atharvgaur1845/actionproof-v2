#!/bin/bash
# Run cluster queue tasks (slurm/tasks/<name>.sh) unchanged on an HF Jobs GPU, then upload their outputs.
#   env: AP_REPO (private dataset repo with bundle/), AP_TASKS ("161_bank_smoke 162_f8_natural_gen ..."),
#        AP_INPUTS (repo paths under inputs/ to restore first), AP_OUTPUTS (paths to upload after each task).
set -uo pipefail
W=/work; mkdir -p $W && cd $W
export AP_ROOT=$W AP_HF_HOME=$W/.cache/huggingface AP_HF_OFFLINE=0 UV_CACHE_DIR=$W/.cache/uv
uvx hf download "$AP_REPO" bundle/actionproof.tar.gz bundle/code.tar.gz --repo-type dataset --local-dir /tmp/b
tar xzf /tmp/b/bundle/actionproof.tar.gz -C $W && tar xzf /tmp/b/bundle/code.tar.gz -C $W
[ -n "${AP_INPUTS:-}" ] && uvx hf download "$AP_REPO" --repo-type dataset --include "inputs/*" --local-dir /tmp/in \
  && cp -r /tmp/in/inputs/. $W/
uv venv -q --python 3.12 .venv
grep -v '^-e ' requirements.lock > /tmp/req.txt
uv pip install -q --python .venv -r /tmp/req.txt
uv pip install -q --python .venv --no-deps -e vendor/AgentTune -e vendor/AuditKIT -e vendor/CuratorKIT -e .
mkdir -p logs
for t in $AP_TASKS; do
  echo "=== $t $(date +%T)"
  bash slurm/tasks/$t.sh 2>&1 | grep -v -iE "warn|it/s\]" | tail -25
  echo "=== $t rc=${PIPESTATUS[0]} $(date +%T)"
  for p in $AP_OUTPUTS logs; do [ -e "$p" ] && .venv/bin/hf upload "$AP_REPO" "$p" "outputs/$p" --repo-type dataset --quiet >/dev/null 2>&1; done
done
echo HFJOB_DONE
