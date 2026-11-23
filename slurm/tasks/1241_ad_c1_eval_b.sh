# minutes: 198
# after: 1230_ad_awrbn_s3 1231_ad_awrbn_s4
# AD-C1 evaluation (PREREGISTRATION_AD_C1.md): cw_conf part B + context baselines + R1 retention.
set -uo pipefail
source slurm/claim.sh || exit 0
source slurm/env.sh
.venv/bin/python scripts/ad/freeze_c1.py --check || exit 1
mkdir -p results/ad/conf results/ad/shift logs
[ -f runs/rft_proc/merged/config.json ] || .venv/bin/python -m actionproof.train.merge runs/rft_proc/final \
  CohereLabs/tiny-aya-global > logs/ad_merge.log 2>&1 || exit 1
conf() {  # tag adapter-or-empty
  s=results/ad/conf/$1__cw_conf.summary.json
  [ -f $s ] || .venv/bin/python -m actionproof.ad.evaluate --model runs/rft_proc/merged ${2:+--adapter $2} --tag $1 \
    --split cw_conf --samples 4 --cost 0.05 --out-dir results/ad/conf > logs/$1_conf.log 2>&1 || { tail -20 logs/$1_conf.log; exit 1; }
  grep -v '^INFO\|^WARNING' logs/$1_conf.log | head -1
}
shift_eval() {  # tag adapter split
  s=results/ad/shift/$1__$3.summary.json
  [ -f $s ] || .venv/bin/python scripts/s4_eval.py --model runs/rft_proc/merged --adapter $2 --tag $1 --split $3 \
    --out-dir results/ad/shift > logs/$1_$3.log 2>&1 || { tail -20 logs/$1_$3.log; exit 1; }
  tail -1 logs/$1_$3.log
}
conf ad_nostd_s0 runs/ad_nostd_s0/final
conf ad_nostd_s1 runs/ad_nostd_s1/final
conf ad_nostd_s2 runs/ad_nostd_s2/final
conf ad_nostd_s3 runs/ad_nostd_s3/final
conf ad_nostd_s4 runs/ad_nostd_s4/final
conf ad_awrbn_s0 runs/ad_awrbn_s0/final
conf ad_awrbn_s1 runs/ad_awrbn_s1/final
conf ad_awrbn_s2 runs/ad_awrbn_s2/final
conf ad_awrbn_s3 runs/ad_awrbn_s3/final
conf ad_awrbn_s4 runs/ad_awrbn_s4/final
conf ad_base_warm ""
conf ad_base_a12rv_s0 runs/a12_proc_rv_s0
conf ad_base_rsc_s0 runs/s3_rv_mix_s0/final
source slurm/r1_evalset.sh "ad_grpo_s0=runs/ad_grpo_s0 ad_awrb_s0=runs/ad_awrb_s0 ad_nostd_s0=runs/ad_nostd_s0 ad_awrbn_s0=runs/ad_awrbn_s0"
echo C1_EVAL_OK
