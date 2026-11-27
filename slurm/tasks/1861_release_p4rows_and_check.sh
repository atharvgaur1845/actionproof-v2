# minutes: 40
# (no dependency: 1860 is filed as failed only because its last step is what this task reruns)
# Release data/rft/p4_aug_rows.jsonl (SEL1's replay source, in its registration) to the traces dataset, then rerun the
# last clean-room step in the final clean clone: download the SEL1-registered rows anonymously (with retries) and
# check them against the hashes recorded before training. Network only.
set -uo pipefail
source slurm/claim.sh || exit 0
source slurm/env.sh
unset HF_HUB_OFFLINE
.venv/bin/python scripts/release_hf.py --execute --only dataset > logs/release_p4rows.log 2>&1 || { tail -30 logs/release_p4rows.log; exit 1; }
C=/scratch/pabitra/ap_cleanroom/final
cd "$C/clone" || exit 1
export HF_HOME="$C/hf" HF_HUB_OFFLINE=0
unset HF_TOKEN HUGGING_FACE_HUB_TOKEN
for i in 1 2 3 4 5; do
  "$C/venv/bin/python" -c "from huggingface_hub import snapshot_download; snapshot_download('atharvgaur18/actionproof-traces', repo_type='dataset', local_dir='.', allow_patterns=['data/rft/sel1_*', 'data/rft/p4_aug_rows.jsonl'])" && break
  echo "download attempt $i failed"; sleep 30
done
"$C/venv/bin/python" scripts/freeze_sel1.py --check; rc=$?
echo "freeze_sel1 exit $rc"
echo "git status: $(git status --porcelain --untracked-files=no | wc -l) tracked files changed"
[ $rc -eq 0 ] && echo P4ROWS_AND_CHECK_OK
