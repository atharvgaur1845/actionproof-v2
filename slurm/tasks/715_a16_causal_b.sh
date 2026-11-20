# after: 714_a16_causal_a
# minutes: 300
# A16 causal tests at the dev-chosen layer: R_V seeds 3-4, R_A seeds 0-2, warm start.
set -uo pipefail
source slurm/claim.sh || exit 0
source slurm/env.sh
export MASTER_PORT=29715
.venv/bin/python scripts/freeze_a15.py --check || exit 1
mkdir -p logs results/a15 results/a16 results/eval/a15
[ -f runs/rft_proc/merged/config.json ] || .venv/bin/python -m actionproof.train.merge runs/rft_proc/final CohereLabs/tiny-aya-global > logs/a15_merge.log 2>&1 || { tail -20 logs/a15_merge.log; exit 1; }
.venv/bin/python scripts/a16_mech.py causal --n-test 100 --models rv_s3=runs/a12_proc_rv_s3/final,rv_s4=runs/a12_proc_rv_s4/final,ra_s0=runs/a12_proc_ra_s0/final,ra_s1=runs/a12_proc_ra_s1/final,ra_s2=runs/a12_proc_ra_s2/final,warm=runs/rft_proc/merged > logs/a16_causal_b.log 2>&1 || { tail -30 logs/a16_causal_b.log; exit 1; }
