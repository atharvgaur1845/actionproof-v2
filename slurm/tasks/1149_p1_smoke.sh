# minutes: 40
# P1 smoke (not analysed): arm B seed 0 on 3 dev tasks, greedy + 1 sample, to exercise merge, hook and scoring
set -uo pipefail
source slurm/claim.sh || exit 0
source slurm/env.sh
.venv/bin/python scripts/freeze_p1.py --check || exit 1
[ -f runs/rft_proc/merged/config.json ] || .venv/bin/python -m actionproof.train.merge runs/rft_proc/final CohereLabs/tiny-aya-global > logs/p1_merge_rft.log 2>&1 || exit 1
.venv/bin/python scripts/p1_eval.py --model runs/rft_proc/merged --adapter runs/s3_rv_mix_s0/final --tag B_smoke \
  --split p1_dev_rec --phase smoke --samples 1 --recon --n-tasks 6 > logs/p1_smoke.log 2>&1 || { tail -40 logs/p1_smoke.log; exit 1; }
tail -1 logs/p1_smoke.log
