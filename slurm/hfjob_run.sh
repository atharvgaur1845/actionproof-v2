#!/bin/bash
# Run ONE queue task (slurm/tasks/$AP_TASK.sh) unchanged on an HF Jobs GPU (Stage 2), then upload its outputs.
# The launcher (scripts/hf_launch.sh) already holds the cross-pool claim, so the task's own claim step is skipped.
#   env: AP_REPO (private dataset repo), AP_TASK, AP_OUTPUTS (paths uploaded under outputs/ after the task).
set -uo pipefail
W=/work; mkdir -p $W && cd $W
export AP_TASKS=${AP_TASKS:-$AP_TASK} AP_HF=1 AP_ROOT=$W AP_HF_HOME=$W/.cache/huggingface AP_HF_OFFLINE=0 UV_CACHE_DIR=$W/.cache/uv
uvx hf download "$AP_REPO" bundle/actionproof.tar.gz "bundle/code_$AP_TASK.tar.gz" --repo-type dataset --local-dir /tmp/b
# --no-same-owner: as root, tar otherwise tries to restore cluster uids and exits non-zero (A11 batches, 2026-11-20)
tar --no-same-owner -xzf /tmp/b/bundle/actionproof.tar.gz -C $W
tar --no-same-owner -xzf "/tmp/b/bundle/code_$AP_TASK.tar.gz" -C $W
test -f "slurm/tasks/$AP_TASK.sh" || { echo "bundle missing task script"; echo 2 > /tmp/rc; uvx hf upload "$AP_REPO" /tmp/rc "outputs/$AP_TASK/RC" --repo-type dataset --quiet 2>/dev/null; exit 2; }
uvx hf download "$AP_REPO" --repo-type dataset --include "inputs/$AP_TASK/*" --local-dir /tmp/in >/dev/null 2>&1 \
  && [ -d "/tmp/in/inputs/$AP_TASK" ] && cp -r "/tmp/in/inputs/$AP_TASK/." $W/
uv venv -q --python 3.12 .venv
grep -v '^-e ' requirements.lock > /tmp/req.txt
uv pip install -q --python .venv -r /tmp/req.txt
uv pip install -q --python .venv --no-deps -e vendor/AgentTune -e vendor/AuditKIT -e vendor/CuratorKIT -e .
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
mkdir -p logs
up() { for p in $AP_OUTPUTS logs; do [ -e "$p" ] && .venv/bin/hf upload "$AP_REPO" "$p" "outputs/$AP_TASK/$p" --repo-type dataset --quiet >/dev/null 2>&1; done; }
( while sleep 1200; do up; done ) &   # periodic upload: a timeout never loses finished checkpoints
rc=0
for t in ${AP_TASKS//,/ }; do
  echo "=== $t $(date +%T)"
  bash "slurm/tasks/$t.sh" 2>&1 | grep -v -iE "warn|it/s\]" | tail -40
  r=${PIPESTATUS[0]}; [ "$r" = 0 ] || rc=$r
  echo "=== $t rc=$r $(date +%T)"
  echo "$r" > /tmp/rc_$t && .venv/bin/hf upload "$AP_REPO" /tmp/rc_$t "outputs/$AP_TASK/RC_$t" --repo-type dataset --quiet >/dev/null 2>&1
  up
done
up
echo "$rc" > /tmp/rc && .venv/bin/hf upload "$AP_REPO" /tmp/rc "outputs/$AP_TASK/RC" --repo-type dataset --quiet >/dev/null 2>&1
echo HFJOB_DONE
