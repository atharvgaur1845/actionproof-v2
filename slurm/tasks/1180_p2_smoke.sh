# minutes: 60
# P2 (PREREGISTRATION_P2.md): smoke (not analysed): tests, ledger and guard on 8 dev tasks, harness equivalence with P1
set -uo pipefail
source slurm/claim.sh || exit 0
source slurm/env.sh
.venv/bin/python scripts/freeze_p2.py --check || exit 1
.venv/bin/python -m pytest -q tests/p2 tests/p1 || exit 1
.venv/bin/python scripts/p2_eval.py --model runs/rft_proc/merged --adapter runs/s3_rv_mix_s0/final --tag B2_smoke --split p1_dev_rec --phase smoke --samples 1 --ledger --n-tasks 8 > logs/p2_smoke_b2.log 2>&1 || { tail -30 logs/p2_smoke_b2.log; exit 1; }
.venv/bin/python scripts/p2_eval.py --model runs/rft_proc/merged --adapter runs/s3_rv_mix_s0/final --tag Ag_smoke --split p1_dev_rec --phase smoke --samples 1 --guard --n-tasks 8 > logs/p2_smoke_ag.log 2>&1 || { tail -30 logs/p2_smoke_ag.log; exit 1; }
.venv/bin/python scripts/p2_eval.py --model runs/rft_proc/merged --adapter runs/s3_rv_mix_s0/final --tag A_equiv --split p1_dev_rec --phase smoke --samples 4 > logs/p2_smoke_eq.log 2>&1 || { tail -30 logs/p2_smoke_eq.log; exit 1; }
.venv/bin/python scripts/p2_equivalence.py > logs/p2_equivalence.log 2>&1; cat logs/p2_equivalence.log
