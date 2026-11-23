# minutes: 120
# P2 (PREREGISTRATION_P2.md): shift splits once (Hindi, Hinglish, renamed API): C, three seeds
set -uo pipefail
source slurm/claim.sh || exit 0
source slurm/env.sh
.venv/bin/python scripts/freeze_p2.py --check || exit 1
[ -f runs/s3_rv_mix_s0/merged/config.json ] || .venv/bin/python -m actionproof.train.merge runs/s3_rv_mix_s0/final runs/rft_proc/merged > logs/p2_merge_rsc0.log 2>&1 || exit 1
[ -f runs/s3_rv_mix_s1/merged/config.json ] || .venv/bin/python -m actionproof.train.merge runs/s3_rv_mix_s1/final runs/rft_proc/merged > logs/p2_merge_rsc1.log 2>&1 || exit 1
[ -f runs/s3_rv_mix_s2/merged/config.json ] || .venv/bin/python -m actionproof.train.merge runs/s3_rv_mix_s2/final runs/rft_proc/merged > logs/p2_merge_rsc2.log 2>&1 || exit 1
source slurm/p2_evalset.sh shift "C_s0|runs/s3_rv_mix_s0/merged|runs/p1_c_s0/final|-" "s4_lang_hi s4_lang_hinglish s4_api"
source slurm/p2_evalset.sh shift "C_s1|runs/s3_rv_mix_s1/merged|runs/p1_c_s1/final|-" "s4_lang_hi s4_lang_hinglish s4_api"
source slurm/p2_evalset.sh shift "C_s2|runs/s3_rv_mix_s2/merged|runs/p1_c_s2/final|-" "s4_lang_hi s4_lang_hinglish s4_api"
