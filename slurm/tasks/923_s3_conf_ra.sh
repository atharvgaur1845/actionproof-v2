# minutes: 120
# S3 confirmatory evaluation: A12 R_A seeds 0-2 (reference)
set -uo pipefail
source slurm/claim.sh || exit 0
source slurm/env.sh
export MASTER_PORT=29923
source slurm/s3_conf.sh "a12_ra_s0=runs/a12_proc_ra_s0/final a12_ra_s1=runs/a12_proc_ra_s1/final a12_ra_s2=runs/a12_proc_ra_s2/final "
