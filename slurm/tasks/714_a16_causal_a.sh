# minutes: 300
# A16 layer choice on dev (fixed rule) then causal tests on test (PREREGISTRATION_A15 §4.3-4.4): R_V seeds 0-2.
set -uo pipefail
source slurm/claim.sh || exit 0
source slurm/env.sh
export MASTER_PORT=29714
.venv/bin/python scripts/freeze_a15.py --check || exit 1
mkdir -p logs results/a15 results/a16 results/eval/a15
[ -f runs/rft_proc/merged/config.json ] || .venv/bin/python -m actionproof.train.merge runs/rft_proc/final CohereLabs/tiny-aya-global > logs/a15_merge.log 2>&1 || { tail -20 logs/a15_merge.log; exit 1; }
.venv/bin/python scripts/a16_mech.py sweep --n-pairs 30 --models rv_s0=runs/a12_proc_rv_s0/final > logs/a16_sweep.log 2>&1 || { tail -30 logs/a16_sweep.log; exit 1; }
.venv/bin/python scripts/a16_mech.py causal --n-test 100 --models rv_s0=runs/a12_proc_rv_s0/final,rv_s1=runs/a12_proc_rv_s1/final,rv_s2=runs/a12_proc_rv_s2/final > logs/a16_causal_a.log 2>&1 || { tail -30 logs/a16_causal_a.log; exit 1; }
