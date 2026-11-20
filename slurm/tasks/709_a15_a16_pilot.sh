# minutes: 60
# A15/A16 pilot (development split only; disclosed, not analysed): one model through the A15 ratio forks on 10
# proc_mech_dev tasks (parsing, throughput), and the A16 hook check, margins, probes and a 3-pair layer sweep on dev.
set -uo pipefail
source slurm/claim.sh || exit 0
source slurm/env.sh
mkdir -p logs results/a15/pilot results/a16/pilot
[ -f runs/rft_proc/merged/config.json ] || .venv/bin/python -m actionproof.train.merge runs/rft_proc/final \
  CohereLabs/tiny-aya-global > logs/a15p_merge.log 2>&1 || { tail -20 logs/a15p_merge.log; exit 1; }
.venv/bin/python -m pytest -q tests/integrity/test_a15.py 2>&1 | tail -3
.venv/bin/python scripts/a15_eval.py model --model runs/rft_proc/merged --adapter runs/a12_proc_rv_s0/final \
  --tag rv_s0 --split proc_mech_dev --n-tasks 10 --out-dir results/a15/pilot > logs/a15p.log 2>&1 || { tail -30 logs/a15p.log; exit 1; }
tail -12 logs/a15p.log
rm -rf runs/a12_proc_rv_s0/merged
.venv/bin/python scripts/a16_mech.py trajectory --models warm=runs/rft_proc/merged,rv_s0=runs/a12_proc_rv_s0/final \
  --n-test 2 --out-dir results/a16/pilot > logs/a16p_traj.log 2>&1 || { tail -30 logs/a16p_traj.log; exit 1; }
tail -3 logs/a16p_traj.log
.venv/bin/python scripts/a16_mech.py sweep --models rv_s0=runs/a12_proc_rv_s0/final --n-pairs 3 \
  --out-dir results/a16/pilot > logs/a16p_sweep.log 2>&1 || { tail -30 logs/a16p_sweep.log; exit 1; }
tail -3 logs/a16p_sweep.log
