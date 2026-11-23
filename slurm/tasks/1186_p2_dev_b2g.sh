# minutes: 90
# P2 (PREREGISTRATION_P2.md): development (descriptive): B2 and Ag, three seeds
set -uo pipefail
source slurm/claim.sh || exit 0
source slurm/env.sh
.venv/bin/python scripts/freeze_p2.py --check || exit 1
[ -f runs/s3_rv_mix_s0/merged/config.json ] || .venv/bin/python -m actionproof.train.merge runs/s3_rv_mix_s0/final runs/rft_proc/merged > logs/p2_merge_rsc0.log 2>&1 || exit 1
[ -f runs/s3_rv_mix_s1/merged/config.json ] || .venv/bin/python -m actionproof.train.merge runs/s3_rv_mix_s1/final runs/rft_proc/merged > logs/p2_merge_rsc1.log 2>&1 || exit 1
[ -f runs/s3_rv_mix_s2/merged/config.json ] || .venv/bin/python -m actionproof.train.merge runs/s3_rv_mix_s2/final runs/rft_proc/merged > logs/p2_merge_rsc2.log 2>&1 || exit 1
source slurm/p2_evalset.sh dev "B2_s0|runs/rft_proc/merged|runs/s3_rv_mix_s0/final|L" "p1_dev_full p1_dev_rec"
source slurm/p2_evalset.sh dev "Ag_s0|runs/rft_proc/merged|runs/s3_rv_mix_s0/final|G" "p1_dev_full p1_dev_rec"
source slurm/p2_evalset.sh dev "B2_s1|runs/rft_proc/merged|runs/s3_rv_mix_s1/final|L" "p1_dev_full p1_dev_rec"
source slurm/p2_evalset.sh dev "Ag_s1|runs/rft_proc/merged|runs/s3_rv_mix_s1/final|G" "p1_dev_full p1_dev_rec"
source slurm/p2_evalset.sh dev "B2_s2|runs/rft_proc/merged|runs/s3_rv_mix_s2/final|L" "p1_dev_full p1_dev_rec"
source slurm/p2_evalset.sh dev "Ag_s2|runs/rft_proc/merged|runs/s3_rv_mix_s2/final|G" "p1_dev_full p1_dev_rec"
