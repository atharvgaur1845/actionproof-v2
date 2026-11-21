# minutes: 200
# S3 addendum 2 Part B: rebalanced recovery-start recipe, seed 1.
set -uo pipefail
source slurm/claim.sh || exit 0
source slurm/env.sh
export MASTER_PORT=29963
.venv/bin/python scripts/freeze_s3a2.py --check || exit 1
source slurm/s3_arm.sh rb_mix rv data/bench/proc/s3_trainb_half_rb.jsonl,data/bench/proc/s3_rec_train_half_rb.jsonl 1 200
