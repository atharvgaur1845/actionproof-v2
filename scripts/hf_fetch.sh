#!/bin/bash
# Pull an HF Jobs task's outputs into the cluster tree and mark it done for the queue: the claim gets a done file and
# a copy of the task goes to queue/h200/done/, which satisfies `# after:` dependencies of later tasks.
#   scripts/hf_fetch.sh <task>
set -euo pipefail
cd /scratch/pabitra/actionproof
TASK=$1; REPO=atharvgaur18/actionproof-jobs
export HF_HOME=/scratch/pabitra/.cache/huggingface HF_HUB_OFFLINE=0
rm -rf .cache/hfjob/dl_$TASK
.venv/bin/hf download $REPO --repo-type dataset --include "outputs/$TASK/*" --local-dir .cache/hfjob/dl_$TASK --quiet
RC=$(cat .cache/hfjob/dl_$TASK/outputs/$TASK/RC 2>/dev/null || echo missing)
rsync -a --exclude 'RC*' .cache/hfjob/dl_$TASK/outputs/$TASK/ ./
echo "fetched $TASK rc=$RC"
for f in .cache/hfjob/dl_$TASK/outputs/$TASK/RC_*; do  # per-task exit codes of a multi-task job
  [ -e "$f" ] || continue; t=${f##*/RC_}; r=$(cat $f)
  if [ "$r" = 0 ]; then date -Is > queue/claims/$t/done; cp slurm/tasks/$t.sh queue/h200/done/$t.sh; rm -f queue/*/pending/$t.sh; fi
  echo "  $t rc=$r"
done
if [ "$RC" = 0 ]; then
  date -Is > queue/claims/$TASK/done
  cp slurm/tasks/$TASK.sh queue/h200/done/$TASK.sh
  rm -f queue/*/pending/$TASK.sh
else
  echo "job rc=$RC; claims of tasks without rc=0 kept until reviewed"
fi
