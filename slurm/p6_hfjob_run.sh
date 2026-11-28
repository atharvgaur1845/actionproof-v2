#!/bin/bash
# Run one P6 (arm, seed) on an HF Jobs GPU: the train task, then the eval task of the same arm and seed, unchanged.
# Same environment as slurm/hfjob_run.sh (bundle/actionproof.tar.gz + requirements.lock); the P6 code, registration,
# P4 adapters and P4 baseline outputs come in one shared bundle/code_p6.tar.gz so every job checks the same snapshot.
#   env: AP_REPO (private dataset repo), AP_TASKS (comma list), AP_TAG (output prefix), AP_OUTPUTS (paths to upload)
set -uo pipefail
W=/work; mkdir -p $W && cd $W
export AP_HF=1 AP_ROOT=$W AP_HF_HOME=$W/.cache/huggingface AP_HF_OFFLINE=0 UV_CACHE_DIR=$W/.cache/uv
uvx hf download "$AP_REPO" bundle/actionproof.tar.gz bundle/code_p6.tar.gz --repo-type dataset --local-dir /tmp/b
tar --no-same-owner -xzf /tmp/b/bundle/actionproof.tar.gz -C $W
tar --no-same-owner -xzf /tmp/b/bundle/code_p6.tar.gz -C $W
uv venv -q --python 3.12 .venv
grep -v '^-e ' requirements.lock > /tmp/req.txt
uv pip install -q --python .venv -r /tmp/req.txt
uv pip install -q --python .venv --no-deps -e vendor/AgentTune -e vendor/AuditKIT -e vendor/CuratorKIT -e .
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
mkdir -p logs
up() { for p in $AP_OUTPUTS logs; do [ -e "$p" ] && .venv/bin/hf upload "$AP_REPO" "$p" "outputs/$AP_TAG/$p" --repo-type dataset --quiet >/dev/null 2>&1; done; }
( while sleep 1200; do up; done ) &
rc=0
for t in ${AP_TASKS//,/ }; do
  echo "=== $t $(date +%T)"
  bash "slurm/tasks/$t.sh" 2>&1 | grep -v -iE "warn|it/s\]" | tail -40
  r=${PIPESTATUS[0]}; [ "$r" = 0 ] || rc=$r
  echo "=== $t rc=$r $(date +%T)"
  echo "$r" > /tmp/rc_$t && .venv/bin/hf upload "$AP_REPO" /tmp/rc_$t "outputs/$AP_TAG/RC_$t" --repo-type dataset --quiet >/dev/null 2>&1
  up
  [ "$r" = 0 ] || break
done
up
echo "$rc" > /tmp/rc && .venv/bin/hf upload "$AP_REPO" /tmp/rc "outputs/$AP_TAG/RC" --repo-type dataset --quiet >/dev/null 2>&1
echo HFJOB_DONE
