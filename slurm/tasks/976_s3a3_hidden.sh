# minutes: 120
# after: 975_s3a3_conf 966_s3a2_conf_matched
# S3 addendum 3 secondary: recovery starts with the history hidden, every arm (FULL, STATE, HIST), seeds 0-4.
set -uo pipefail
source slurm/claim.sh || exit 0
source slurm/env.sh
export MASTER_PORT=29976
.venv/bin/python scripts/freeze_s3a3.py --check || exit 1
mkdir -p logs results/s3/hidden
for p in s3_matched s3_stateonly s3_rv_mix; do for s in 0 1 2 3 4; do
  tag=${p}_s$s; o=results/s3/hidden/${tag}__s3_conf_rec.summary.json
  [ -f $o ] || .venv/bin/python scripts/s3_eval_hidden.py --model runs/rft_proc/merged --adapter runs/$tag/final \
    --tag $tag --split s3_conf_rec --out-dir results/s3/hidden > logs/s3h_$tag.log 2>&1 || { tail -20 logs/s3h_$tag.log; exit 1; }
  rm -rf runs/$tag/merged
done; done
