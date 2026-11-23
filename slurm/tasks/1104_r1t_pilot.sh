# minutes: 60
# R1-T τ-bench (PREREGISTRATION_R1_TAU.md): feasibility pilot, base, 20 tasks, 1 trial
set -uo pipefail
source slurm/claim.sh || exit 0
source slurm/env.sh
R1T_LIMIT=20 R1T_TRIALS=1 R1T_OUT=results/r1_tau/pilot source slurm/r1_tau_set.sh "base="
