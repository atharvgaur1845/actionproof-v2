# minutes: 200
# S3 confirmatory evaluation: warm start and A12 R_V seeds 0-4
set -uo pipefail
source slurm/claim.sh || exit 0
source slurm/env.sh
export MASTER_PORT=29920
source slurm/s3_conf.sh "warm_start= a12_rv_s0=runs/a12_proc_rv_s0/final a12_rv_s1=runs/a12_proc_rv_s1/final a12_rv_s2=runs/a12_proc_rv_s2/final a12_rv_s3=runs/a12_proc_rv_s3/final a12_rv_s4=runs/a12_proc_rv_s4/final "
