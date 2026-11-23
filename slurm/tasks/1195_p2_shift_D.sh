# minutes: 120
# P2 (PREREGISTRATION_P2.md): shift splits once (Hindi, Hinglish, renamed API): D, three seeds
# after: 1154_p1_d_s0 1155_p1_d_s1 1156_p1_d_s2
set -uo pipefail
source slurm/claim.sh || exit 0
source slurm/env.sh
.venv/bin/python scripts/freeze_p2.py --check || exit 1
source slurm/p2_evalset.sh shift "D_s0|runs/p1_dwarm/merged|runs/p1_d_s0/final|-" "s4_lang_hi s4_lang_hinglish s4_api"
source slurm/p2_evalset.sh shift "D_s1|runs/p1_dwarm/merged|runs/p1_d_s1/final|-" "s4_lang_hi s4_lang_hinglish s4_api"
source slurm/p2_evalset.sh shift "D_s2|runs/p1_dwarm/merged|runs/p1_d_s2/final|-" "s4_lang_hi s4_lang_hinglish s4_api"
