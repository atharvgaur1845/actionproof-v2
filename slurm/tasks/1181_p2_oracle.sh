# minutes: 30
# P2 (PREREGISTRATION_P2.md): scripted reference policy on test and language-shift splits (CPU)
set -uo pipefail
source slurm/claim.sh || exit 0
source slurm/env.sh
.venv/bin/python scripts/freeze_p2.py --check || exit 1
.venv/bin/python scripts/p2_oracle.py
