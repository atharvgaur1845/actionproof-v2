#!/bin/bash
#SBATCH -p compute
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH -J sh_diag
#SBATCH --output=/scratch/pabitra/selfheal/logs/%x-%j.out
# DIAGNOSE (descriptive): failure taxonomy of existing released-agent episodes. Reads $R only.
set -euo pipefail
source /scratch/pabitra/selfheal/wt/slurm/selfheal/common.sh
O=$SH/results/diagnose
d() { out=$1; shift; $PY scripts/selfheal_run.py diagnose --out "$O/$out" --inputs "$@" > "$SH/logs/diag_$out.log" 2>&1 || { tail -30 "$SH/logs/diag_$out.log"; exit 1; }; echo "done $out"; }
d txn_test $R/txn/D_s{0,1,2}__txn_test.jsonl $R/txn/Dwaug_s{0,1,2}__txn_test.jsonl $R/txn/B4__txn_test.jsonl
d txn_dev $R/rg1/dev_D_s0_none__txn_dev.jsonl
d p1_dev $R/p1/dev/D_s{0,1,2}__p1_dev_full.jsonl $R/p1/dev/D_s{0,1,2}__p1_dev_rec.jsonl
d p1_test $R/p1/test/D_s{0,1,2}__p1_test_{full,pp,rec,wf4,combo}.jsonl $R/p4/Dwaug_s{0,1,2}__p1_test_{full,pp,rec,wf4,combo}.jsonl
echo SH_DIAGNOSE_OK
