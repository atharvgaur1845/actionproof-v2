# minutes: 120
# P1 practical agent (PREREGISTRATION_P1.md): arms A and B (RSC seeds 0-2, without and with the inference-time reconciliation line), dev evaluation
set -uo pipefail
source slurm/claim.sh || exit 0
source slurm/env.sh
.venv/bin/python scripts/freeze_p1.py --check || exit 1
source slurm/p1_evalset.sh dev "A_s0|runs/rft_proc/merged|runs/s3_rv_mix_s0/final|0 A_s1|runs/rft_proc/merged|runs/s3_rv_mix_s1/final|0 A_s2|runs/rft_proc/merged|runs/s3_rv_mix_s2/final|0 B_s0|runs/rft_proc/merged|runs/s3_rv_mix_s0/final|1 B_s1|runs/rft_proc/merged|runs/s3_rv_mix_s1/final|1 B_s2|runs/rft_proc/merged|runs/s3_rv_mix_s2/final|1" "p1_dev_full p1_dev_rec"
