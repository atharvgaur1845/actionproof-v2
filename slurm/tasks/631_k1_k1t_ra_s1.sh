# minutes: 330
# K1 R_A GRPO on proc_train_k1t, seed 1 (PREREGISTRATION_K1).
set -uo pipefail
source slurm/claim.sh || exit 0
source slurm/env.sh
export MASTER_PORT=29631  # two Slurm lanes share a node
source slurm/k1_arm.sh k1t 1
