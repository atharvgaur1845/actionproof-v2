# minutes: 90
# after: 1820_release_sel1
# Phase F clean-room reproduction. A fresh clone of the sprint branch from a git bundle (sha256 checked), a fresh
# virtualenv with only the documented dependencies, an anonymous Hugging Face cache (no token), then
#   python scripts/smoke_check.py   and   bash scripts/verify_numbers.sh
# Never touches the main working copy's results/. Log: /scratch/pabitra/ap_cleanroom/cleanroom.log
set -uo pipefail
source slurm/claim.sh || exit 0
C=/scratch/pabitra/ap_cleanroom
BUNDLE_SHA=2d749607a90e13aaf426980d742e81b8f99c2c94c4dfdf9e7a26466cb81578b1
PYBASE=/home/pabitra/.local/share/uv/python/cpython-3.12.15-linux-x86_64-gnu/bin/python3.12
echo "$(sha256sum $C/actionproof.bundle | cut -c1-64)" | grep -q "$BUNDLE_SHA" || { echo "bundle hash mismatch"; exit 1; }
rm -rf "$C/clone" "$C/venv" "$C/hf"
git clone -q --branch stage2-superiority-sprint "$C/actionproof.bundle" "$C/clone" || exit 1
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
  echo "git status after: $(git status --porcelain --untracked-files=no | wc -l) tracked files changed"
} > "$C/cleanroom.log" 2>&1
tail -45 "$C/cleanroom.log"
grep -q "smoke_check exit 0" "$C/cleanroom.log" && grep -q "verify_numbers exit 0" "$C/cleanroom.log" && echo CLEANROOM_OK
