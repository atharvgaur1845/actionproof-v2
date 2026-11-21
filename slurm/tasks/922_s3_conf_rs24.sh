# minutes: 200
# after: 912_s3_rs_s2 913_s3_rs_s3 914_s3_rs_s4
# S3 confirmatory evaluation: RS-GRPO seeds 2-4
set -uo pipefail
source slurm/claim.sh || exit 0
source slurm/env.sh
export MASTER_PORT=29922
source slurm/s3_conf.sh "s3_rv_mix_s2=runs/s3_rv_mix_s2/final s3_rv_mix_s3=runs/s3_rv_mix_s3/final s3_rv_mix_s4=runs/s3_rv_mix_s4/final"
