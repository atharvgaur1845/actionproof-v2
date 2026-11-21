# minutes: 150
# after: 970_s3a3_stateonly_s0 971_s3a3_stateonly_s1 972_s3a3_stateonly_s2 973_s3a3_stateonly_s3 974_s3a3_stateonly_s4
# S3 addendum 3: state-only models once on every S3 confirmatory split.
set -uo pipefail
source slurm/claim.sh || exit 0
source slurm/env.sh
export MASTER_PORT=29975
.venv/bin/python scripts/freeze_s3a3.py --check || exit 1
source slurm/s3_conf.sh "s3_stateonly_s0=runs/s3_stateonly_s0/final s3_stateonly_s1=runs/s3_stateonly_s1/final s3_stateonly_s2=runs/s3_stateonly_s2/final s3_stateonly_s3=runs/s3_stateonly_s3/final s3_stateonly_s4=runs/s3_stateonly_s4/final"
