# minutes: 120
# Stage 3 pilot (development only): safe reward, original pool
set -uo pipefail
source slurm/claim.sh || exit 0
source slurm/env.sh
export MASTER_PORT=29901
source slurm/s3_arm.sh p_safe_b safe data/bench/proc/proc_train_b.jsonl 0 100
