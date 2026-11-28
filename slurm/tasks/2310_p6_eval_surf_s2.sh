# minutes: 120
# after: 2300_p6_surf_s2
set -euo pipefail
source slurm/claim.sh || exit 0
source slurm/env.sh
source slurm/p6_eval_common.sh
run_p6_eval surface 2
