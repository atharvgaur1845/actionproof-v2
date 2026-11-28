# minutes: 90
# AD-C2 (PREREGISTRATION_AD_C2.md; runner fix only, PREREGISTRATION_AD_C2_ADDENDUM2.md): evaluate the five seed-0 models on sealed cw_c2 once, then run the registered analysis once.
set -uo pipefail
source slurm/claim.sh || exit 0
source slurm/env.sh
export MASTER_PORT=29303
.venv/bin/python scripts/ad/freeze_c2.py --check || exit 1
mkdir -p results/ad/c2 logs
[ -f runs/rft_proc/merged/config.json ] || .venv/bin/python -m actionproof.train.merge runs/rft_proc/final CohereLabs/tiny-aya-global > logs/ad_merge.log 2>&1 || exit 1
ev() {  # tag adapter
  [ -f results/ad/c2/$1__cw_c2.summary.json ] || .venv/bin/python -m actionproof.ad.evaluate --model runs/rft_proc/merged --adapter $2 \
    --tag $1 --split cw_c2 --samples 4 --cost 0.05 --out-dir results/ad/c2 > logs/$1_c2.log 2>&1 || { tail -20 logs/$1_c2.log; exit 1; }
  grep -v "^INFO\|^WARNING" logs/$1_c2.log | head -1 || true  # without || true, SIGPIPE + pipefail failed task 2003
}
for t in ad_grpo_s0 ad_nostd_s0 ad_c2_grpo_c001_s0 ad_c2_nostd_c001_s0 ad_c2_grpo_c000_s0; do
  [ -f runs/$t/final/adapter_config.json ] || { echo "missing runs/$t/final"; exit 1; }
done
for t in ad_grpo_s0 ad_nostd_s0 ad_c2_grpo_c001_s0 ad_c2_nostd_c001_s0 ad_c2_grpo_c000_s0; do ev $t runs/$t/final || exit 1; done
.venv/bin/python scripts/ad/c2_stats.py > logs/c2_stats.log 2>&1 || { tail -20 logs/c2_stats.log; exit 1; }
cat results/ad/c2/c2_summary.md
