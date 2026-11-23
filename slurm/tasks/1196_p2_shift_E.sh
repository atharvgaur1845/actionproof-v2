# minutes: 120
# P2 (PREREGISTRATION_P2.md): shift splits once (Hindi, Hinglish, renamed API): E, three seeds
# after: 1183_p2_e_s0 1184_p2_e_s1 1185_p2_e_s2
set -uo pipefail
source slurm/claim.sh || exit 0
source slurm/env.sh
.venv/bin/python scripts/freeze_p2.py --check || exit 1
source slurm/p2_evalset.sh shift "E_s0|runs/p2_ewarm/merged|runs/p2_e_s0/final|L" "s4_lang_hi s4_lang_hinglish s4_api"
source slurm/p2_evalset.sh shift "E_s1|runs/p2_ewarm/merged|runs/p2_e_s1/final|L" "s4_lang_hi s4_lang_hinglish s4_api"
source slurm/p2_evalset.sh shift "E_s2|runs/p2_ewarm/merged|runs/p2_e_s2/final|L" "s4_lang_hi s4_lang_hinglish s4_api"
