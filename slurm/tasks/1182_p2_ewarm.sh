# minutes: 90
# P2 (PREREGISTRATION_P2.md): arm E warm start: arm D's rows with the ledger, D's exact SFT settings
set -uo pipefail
source slurm/claim.sh || exit 0
source slurm/env.sh
.venv/bin/python scripts/freeze_p2.py --check || exit 1
[ -f data/rft/p2_ewarm_rows.stats.json ] || .venv/bin/python -m actionproof.p2.build || exit 1
[ -f runs/p2_ewarm/final/adapter_config.json ] || .venv/bin/python -m actionproof.train.rft --data data/rft/p2_ewarm_rows.jsonl \
  --out runs/p2_ewarm --epochs 2 --seed 0 > logs/p2_ewarm.full.log 2>&1 || { tail -30 logs/p2_ewarm.full.log; exit 1; }
rm -rf runs/p2_ewarm/checkpoint-*/optimizer.pt
[ -f runs/p2_ewarm/merged/config.json ] || .venv/bin/python -m actionproof.train.merge runs/p2_ewarm/final CohereLabs/tiny-aya-global > logs/p2_merge_ewarm.log 2>&1 || exit 1
