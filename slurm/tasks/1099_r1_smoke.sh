# minutes: 30
# R1 capability retention (PREREGISTRATION_R1.md): smoke test, base model, 20 inputs per benchmark, not analysed
set -uo pipefail
source slurm/claim.sh || exit 0
source slurm/env.sh
R1_LIMIT=20 R1_OUT=results/r1/smoke source slurm/r1_evalset.sh "base="
