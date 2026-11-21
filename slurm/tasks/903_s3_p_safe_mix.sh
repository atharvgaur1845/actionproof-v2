# minutes: 120
# Stage 3 pilot (development only): safe reward, recovery-start mix
set -uo pipefail
source slurm/claim.sh || exit 0
source slurm/env.sh
export MASTER_PORT=29903
source slurm/s3_arm.sh p_safe_mix safe data/bench/proc/s3_trainb_half.jsonl,data/bench/proc/s3_rec_train_half.jsonl 0 100
