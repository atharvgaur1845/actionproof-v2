#!/bin/bash
#SBATCH -p compute
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH -J sh_corr
#SBATCH --output=/scratch/pabitra/selfheal/logs/%x-%j.out
# CORRECT with the B4 teacher on DEV-split failures only (CPU simulator replay), then a pilot curation.
# Round label r0_dev: a pipeline pilot on dev data, not a registered loop round.
set -euo pipefail
source /scratch/pabitra/selfheal/wt/slurm/selfheal/common.sh
O=$SH/results/r0_dev
L=$SH/results/ledger.json
mkdir -p $O
# fresh splits (built once) and gate registration, so no gate task can become correction data
[ -f data/bench/proc/selfheal_manifest.json ] || $PY scripts/selfheal_run.py splits > $SH/logs/splits.log 2>&1 || { tail -30 $SH/logs/splits.log; exit 1; }
$PY scripts/selfheal_run.py splits --check || exit 1
$PY scripts/selfheal_run.py register-gate --ledger $L || exit 1
$PY scripts/selfheal_run.py correct --teacher b4 --ledger $L --round r0_dev --out $O/corrections_p1_dev.jsonl \
  --inputs $R/p1/dev/D_s{0,1,2}__p1_dev_full.jsonl $R/p1/dev/D_s{0,1,2}__p1_dev_rec.jsonl > $SH/logs/corr_p1.log 2>&1 || { tail -30 $SH/logs/corr_p1.log; exit 1; }
$PY scripts/selfheal_run.py correct --teacher b4 --ledger $L --round r0_dev --out $O/corrections_txn_dev.jsonl \
  --inputs $R/rg1/dev_D_s0_none__txn_dev.jsonl > $SH/logs/corr_txn.log 2>&1 || { tail -30 $SH/logs/corr_txn.log; exit 1; }
$PY scripts/selfheal_run.py curate --ledger $L --round r0_dev --out-dir $O/curated \
  --corrections $O/corrections_p1_dev.jsonl $O/corrections_txn_dev.jsonl > $SH/logs/curate_r0.log 2>&1 || { tail -30 $SH/logs/curate_r0.log; exit 1; }
tail -40 $SH/logs/curate_r0.log
echo SH_CORRECT_OK
