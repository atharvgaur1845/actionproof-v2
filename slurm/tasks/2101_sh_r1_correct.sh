# minutes: 90
# Self-heal round 1, PREPARED, NOT SUBMITTED: diagnose + B4-teacher corrections + CuratorKIT curation + replay mix (CPU).
# after: 2100_sh_r1_pool_eval
set -uo pipefail
source wt/slurm/claim.sh || exit 0
source wt/slurm/selfheal/round.sh
correct_curate r1 D
echo SH_R1_CORRECT_OK
