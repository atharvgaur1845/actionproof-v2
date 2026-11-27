# minutes: 90
# after: 1851_cards_refresh
# Phase F clean-room reproduction at the final HEAD (319a383), after the hostile-review fixes. Adds: the post-hoc
# validity analysis must recompute results/audit/txn_validity.json byte-identically, and the released SEL1 training
# rows must match the hashes recorded before training.
# Phase F clean-room reproduction. A fresh clone of the sprint branch from a git bundle (sha256 checked), a fresh
# virtualenv with only the documented dependencies, an anonymous Hugging Face cache (no token), then
#   python scripts/smoke_check.py   and   bash scripts/verify_numbers.sh
# Never touches the main working copy's results/. Log: /scratch/pabitra/ap_cleanroom/cleanroom.log
set -uo pipefail
source slurm/claim.sh || exit 0
C=/scratch/pabitra/ap_cleanroom/final
BUNDLE_SHA=afbe0c60a4c4e03e4e4e6f16f65f0eb619b4c1773dc969300d86a15e90a3643c
PYBASE=/home/pabitra/.local/share/uv/python/cpython-3.12.15-linux-x86_64-gnu/bin/python3.12
echo "$(sha256sum /scratch/pabitra/ap_cleanroom/actionproof_final.bundle | cut -c1-64)" | grep -q "$BUNDLE_SHA" || { echo "bundle hash mismatch"; exit 1; }
rm -rf "$C/clone" "$C/venv" "$C/hf"
git clone -q --branch stage2-superiority-sprint "/scratch/pabitra/ap_cleanroom/actionproof_final.bundle" "$C/clone" || exit 1
cd "$C/clone"
echo "clean clone HEAD $(git rev-parse HEAD); status: $(git status --porcelain | wc -l) changed files"
"$PYBASE" -m venv "$C/venv" || exit 1
"$C/venv/bin/pip" install -q --upgrade pip > "$C/pip.log" 2>&1
"$C/venv/bin/pip" install -q -e ".[dev]" huggingface_hub >> "$C/pip.log" 2>&1 || { tail -20 "$C/pip.log"; exit 1; }
"$C/venv/bin/pip" freeze > "$C/pip_freeze.txt"
export HF_HOME="$C/hf" HF_HUB_OFFLINE=0 HF_HUB_DISABLE_TELEMETRY=1
unset HF_TOKEN HUGGING_FACE_HUB_TOKEN TRANSFORMERS_OFFLINE
{
  echo "== smoke_check"; t0=$(date +%s)
  "$C/venv/bin/python" scripts/smoke_check.py; s1=$?
  echo "smoke_check exit $s1 in $(( $(date +%s) - t0 )) s"
  echo "== verify_numbers (anonymous download)"; t0=$(date +%s)
  PYTHON="$C/venv/bin/python" bash scripts/verify_numbers.sh; s2=$?
  echo "verify_numbers exit $s2 in $(( $(date +%s) - t0 )) s"
  echo "== post-hoc validity analysis (exploratory) recomputed"
  "$C/venv/bin/python" scripts/audit/txn_validity.py > /dev/null; s3=$?
  git diff --quiet -- results/audit/ && echo "txn_validity identical (exit $s3)" || echo "txn_validity DIFFERENT (exit $s3)"
  echo "== SEL1 training rows from the public dataset against the pre-training hashes"
  "$C/venv/bin/python" -c "from huggingface_hub import snapshot_download; snapshot_download('atharvgaur18/actionproof-traces', repo_type='dataset', local_dir='.', allow_patterns=['data/rft/sel1_*'])"
  "$C/venv/bin/python" scripts/freeze_sel1.py --check; echo "freeze_sel1 exit $?"
  echo "git status after: $(git status --porcelain --untracked-files=no | wc -l) tracked files changed"
} > "$C/cleanroom.log" 2>&1
tail -45 "$C/cleanroom.log"
grep -q "smoke_check exit 0" "$C/cleanroom.log" && grep -q "verify_numbers exit 0" "$C/cleanroom.log" && grep -q "txn_validity identical" "$C/cleanroom.log" && grep -q "freeze_sel1 exit 0" "$C/cleanroom.log" && echo CLEANROOM_FINAL_OK
