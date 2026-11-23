# minutes: 90
# P1 practical agent (PREREGISTRATION_P1.md): arm D warm start: original warm-start demos + recovery demos, same SFT settings as rft_proc
set -uo pipefail
source slurm/claim.sh || exit 0
source slurm/env.sh
.venv/bin/python scripts/freeze_p1.py --check || exit 1
[ -f data/rft/p1_dwarm_rows.jsonl ] || cat data/rft/proc_demo_rows.jsonl data/rft/p1_recovery_rows.jsonl > data/rft/p1_dwarm_rows.jsonl
[ -f runs/p1_dwarm/final/adapter_config.json ] || .venv/bin/python -m actionproof.train.rft --data data/rft/p1_dwarm_rows.jsonl \
  --out runs/p1_dwarm --epochs 2 --seed 0 > logs/p1_dwarm.full.log 2>&1 || { tail -30 logs/p1_dwarm.full.log; exit 1; }
rm -rf runs/p1_dwarm/checkpoint-*/optimizer.pt
[ -f runs/p1_dwarm/merged/config.json ] || .venv/bin/python -m actionproof.train.merge runs/p1_dwarm/final CohereLabs/tiny-aya-global > logs/p1_merge_dwarm.log 2>&1 || exit 1
