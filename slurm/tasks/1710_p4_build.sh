# minutes: 20
# P4 data (PREREGISTRATION_P4.md): surface- and composition-randomised demonstrations, row-matched to Dw. CPU only.
set -uo pipefail
source slurm/claim.sh || exit 0
source slurm/env.sh
.venv/bin/python -m actionproof.p4.build || exit 1
echo P4_BUILD_OK
