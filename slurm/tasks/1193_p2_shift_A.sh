# minutes: 120
# P2 (PREREGISTRATION_P2.md): shift splits once (Hindi, Hinglish, renamed API): A, three seeds
set -uo pipefail
source slurm/claim.sh || exit 0
source slurm/env.sh
.venv/bin/python scripts/freeze_p2.py --check || exit 1
source slurm/p2_evalset.sh shift "A_s0|runs/rft_proc/merged|runs/s3_rv_mix_s0/final|-" "s4_lang_hi s4_lang_hinglish s4_api"
source slurm/p2_evalset.sh shift "A_s1|runs/rft_proc/merged|runs/s3_rv_mix_s1/final|-" "s4_lang_hi s4_lang_hinglish s4_api"
source slurm/p2_evalset.sh shift "A_s2|runs/rft_proc/merged|runs/s3_rv_mix_s2/final|-" "s4_lang_hi s4_lang_hinglish s4_api"
