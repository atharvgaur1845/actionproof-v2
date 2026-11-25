# minutes: 15
# R7 (exploratory, CPU only): what GRPO adds after recovery SFT (Dw -> D) and how D fails. Moved off the CPU partition (est. start Nov 28).
set -uo pipefail
source slurm/claim.sh || exit 0
source slurm/env.sh
.venv/bin/python scripts/audit/r7_grpo_increment.py || exit 1
echo R7_OK
