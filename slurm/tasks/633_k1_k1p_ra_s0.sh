# minutes: 330
# K1 R_A GRPO on proc_train_k1p, seed 0 (PREREGISTRATION_K1).
set -uo pipefail
source slurm/claim.sh || exit 0
source slurm/env.sh
source slurm/k1_arm.sh k1p 0
