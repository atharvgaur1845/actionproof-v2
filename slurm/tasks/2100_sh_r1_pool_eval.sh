# minutes: 300
# Self-heal round 1 (docs/selfheal/DESIGN.md), PREPARED, NOT SUBMITTED: incumbent D_s0-2 on the round-1 loop pools.
# Runs under slurm/selfheal/worker.sh (ROOT=/scratch/pabitra/selfheal), never in the frozen project.
set -uo pipefail
source wt/slurm/claim.sh || exit 0
source wt/slurm/selfheal/round.sh
pool_eval r1
echo SH_R1_POOL_OK
