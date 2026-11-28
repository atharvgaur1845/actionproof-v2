# minutes: 90
# Final-sprint clean-room reproduction at 5fbc00dc05184f73ba498ebe31616fc549bef0d1, cloned from the public GitHub repository (branch submission-candidate),
# fresh virtualenv with the documented dependencies, anonymous HF cache. smoke_check + verify_numbers.sh (now also
# RG1, RG1 addendum 1 and RG2 from released episodes). Log: /scratch/pabitra/ap_cleanroom/rg/cleanroom.log
set -uo pipefail
source slurm/claim.sh || exit 0
C=/scratch/pabitra/ap_cleanroom/rg
PYBASE=/home/pabitra/.local/share/uv/python/cpython-3.12.15-linux-x86_64-gnu/bin/python3.12
rm -rf "$C" && mkdir -p "$C"
git clone -q --branch submission-candidate https://github.com/atharvgaur1845/actionproof-v2.git "$C/clone" || exit 1
cd "$C/clone"
echo "clean clone HEAD $(git rev-parse HEAD) (expected 5fbc00dc05184f73ba498ebe31616fc549bef0d1)"
"$PYBASE" -m venv "$C/venv" || exit 1
"$C/venv/bin/pip" install -q --upgrade pip > "$C/pip.log" 2>&1
"$C/venv/bin/pip" install -q -e ".[dev]" huggingface_hub >> "$C/pip.log" 2>&1 || { tail -20 "$C/pip.log"; exit 1; }
export HF_HOME="$C/hf" HF_HUB_OFFLINE=0 HF_HUB_DISABLE_TELEMETRY=1
unset HF_TOKEN HUGGING_FACE_HUB_TOKEN TRANSFORMERS_OFFLINE
{
  echo "== smoke_check"; t0=$(date +%s); "$C/venv/bin/python" scripts/smoke_check.py; s1=$?; echo "smoke exit $s1 in $(( $(date +%s) - t0 )) s"
  echo "== verify_numbers"; t0=$(date +%s); PYTHON="$C/venv/bin/python" bash scripts/verify_numbers.sh; s2=$?; echo "verify exit $s2 in $(( $(date +%s) - t0 )) s"
  echo "== tracked files changed: $(git status --porcelain --untracked-files=no | wc -l)"
} > "$C/cleanroom.log" 2>&1
tail -25 "$C/cleanroom.log"
grep -q "smoke exit 0" "$C/cleanroom.log" && grep -q "verify exit 0" "$C/cleanroom.log" && echo CLEANROOM_RG_OK
