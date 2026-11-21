# minutes: 200
# S3 confirmatory evaluation: RS-GRPO seeds 0-1
set -uo pipefail
source slurm/claim.sh || exit 0
source slurm/env.sh
export MASTER_PORT=29921
source slurm/s3_conf.sh "s3_rv_mix_s0=runs/s3_rv_mix_s0/final s3_rv_mix_s1=runs/s3_rv_mix_s1/final"
