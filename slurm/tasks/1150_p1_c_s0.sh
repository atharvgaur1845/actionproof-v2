# minutes: 120
# P1 practical agent (PREREGISTRATION_P1.md): arm C seed 0: recovery SFT on top of RSC seed 0, then dev evaluation
set -uo pipefail
source slurm/claim.sh || exit 0
source slurm/env.sh
.venv/bin/python scripts/freeze_p1.py --check || exit 1
[ -f runs/rft_proc/merged/config.json ] || .venv/bin/python -m actionproof.train.merge runs/rft_proc/final CohereLabs/tiny-aya-global > logs/p1_merge_rft.log 2>&1 || exit 1
[ -f runs/s3_rv_mix_s0/merged/config.json ] || .venv/bin/python -m actionproof.train.merge runs/s3_rv_mix_s0/final runs/rft_proc/merged > logs/p1_merge_rsc0.log 2>&1 || exit 1
[ -f runs/p1_c_s0/final/adapter_config.json ] || .venv/bin/python -m actionproof.train.rft --data data/rft/p1_recovery_rows.jsonl \
  --model runs/s3_rv_mix_s0/merged --out runs/p1_c_s0 --epochs 1 --lr 5e-5 --seed 0 > logs/p1_c_s0.full.log 2>&1 || { tail -30 logs/p1_c_s0.full.log; exit 1; }
rm -rf runs/p1_c_s0/checkpoint-*
source slurm/p1_evalset.sh dev "C_s0|runs/s3_rv_mix_s0/merged|runs/p1_c_s0/final|0" "p1_dev_full p1_dev_rec"
