# minutes: 240
# SR1 (PREREGISTRATION_SR1): RSC + tool-surface randomisation, seed 1.
set -uo pipefail
source slurm/claim.sh || exit 0
source slurm/env.sh
export MASTER_PORT=29988
.venv/bin/python scripts/freeze_sr1.py --check || exit 1
source slurm/s3v_arm.sh sr_mix --surface-random data/bench/proc/s3_trainb_half.jsonl,data/bench/proc/s3_rec_train_half.jsonl 1 200
