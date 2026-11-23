# minutes: 90
# after: 1200_ad_smoke
# Algorithm discovery AD-D1 (docs/research/algorithm_discovery/EXPERIMENT_PROTOCOL.md): ad_baselines.
set -uo pipefail
source slurm/claim.sh || exit 0
source slurm/env.sh
export MASTER_PORT=29210
.venv/bin/python scripts/ad/freeze.py --check || exit 1
mkdir -p results/ad/dev
ev() {  # tag adapter split samples
  s=results/ad/dev/$1__$3.summary.json
  [ -f $s ] || .venv/bin/python -m actionproof.ad.evaluate --model runs/rft_proc/merged ${2:+--adapter $2} --tag $1 \
    --split $3 --n-tasks 150 --samples $4 --out-dir results/ad/dev > logs/ad_eval_$1_$3.log 2>&1 \
    || { tail -20 logs/ad_eval_$1_$3.log; exit 1; }
  grep -v '^INFO\|^WARNING' logs/ad_eval_$1_$3.log | tail -5
}
ev ad_base_warm "" cw_dev 4
ev ad_base_a12rv_s0 runs/a12_proc_rv_s0 cw_dev 4
ev ad_base_rsc_s0 runs/s3_rv_mix_s0/final cw_dev 4
