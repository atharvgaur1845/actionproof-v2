# minutes: 20
# after: 2310_p6_eval_can_s0 2310_p6_eval_can_s1 2310_p6_eval_can_s2 2310_p6_eval_surf_s0 2310_p6_eval_surf_s1 2310_p6_eval_surf_s2
set -euo pipefail
source slurm/claim.sh || exit 0
source slurm/env.sh
.venv/bin/python scripts/freeze_p6.py --check
.venv/bin/python scripts/p6_stats.py
