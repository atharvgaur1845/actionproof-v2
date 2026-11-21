# minutes: 200
# after: 977_sr1_s0 978_sr1_s1 979_sr1_s2
# SR1: surface-randomised models once on the S3 confirmatory splits and the S4 shift splits.
set -uo pipefail
source slurm/claim.sh || exit 0
source slurm/env.sh
export MASTER_PORT=29990
.venv/bin/python scripts/freeze_sr1.py --check || exit 1
M="s3_sr_mix_s0=runs/s3_sr_mix_s0/final s3_sr_mix_s1=runs/s3_sr_mix_s1/final s3_sr_mix_s2=runs/s3_sr_mix_s2/final"
( source slurm/s3_conf.sh "$M" ) || exit 1
source slurm/s4_evalset.sh "$M"
