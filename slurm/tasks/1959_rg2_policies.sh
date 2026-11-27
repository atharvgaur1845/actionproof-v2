# minutes: 15
# RG2 scripted arms on rg2_test (CPU; scripts/rg2_policies_eval.py).
set -uo pipefail
source slurm/claim.sh || exit 0
source slurm/env.sh
.venv/bin/python scripts/rg2_policies_eval.py > logs/rg2_policies.log 2>&1 || { tail -20 logs/rg2_policies.log; exit 1; }
cat logs/rg2_policies.log
echo RG2_POLICIES_OK
