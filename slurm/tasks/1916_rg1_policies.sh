# minutes: 20
# RG1 scripted arms on rg1_test (CPU; scripts/rg1_policies_eval.py): B0-B4, A1-A6 without the gate; B0, B1, B3, B4 with it.
set -uo pipefail
source slurm/claim.sh || exit 0
source slurm/env.sh
mkdir -p logs results/rg1
.venv/bin/python scripts/rg1_policies_eval.py --split rg1_test > logs/rg1_policies.log 2>&1 || { tail -30 logs/rg1_policies.log; exit 1; }
cat logs/rg1_policies.log
echo RG1_POLICIES_OK
