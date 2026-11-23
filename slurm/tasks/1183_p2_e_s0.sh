# minutes: 210
# P2 (PREREGISTRATION_P2.md): arm E seed 0: the RSC recipe unchanged (R_V, s3_trainb_half + s3_rec_train_half, 200 steps) with ledger rollouts
# after: 1182_p2_ewarm
set -uo pipefail
source slurm/claim.sh || exit 0
source slurm/env.sh
.venv/bin/python scripts/freeze_p2.py --check || exit 1
export MASTER_PORT=29960 AP_ROLLOUT_LOG=results/p2/rollouts/E_s0.jsonl
mkdir -p results/p2/rollouts
[ -f runs/p2_e_s0/final/adapter_config.json ] || .venv/bin/python -m actionproof.p2.train --reward rv \
  --pools data/bench/proc/s3_trainb_half.jsonl,data/bench/proc/s3_rec_train_half.jsonl --seed 0 --steps 200 \
  --init-adapter runs/p2_ewarm/final --out runs/p2_e_s0 --use-vllm > logs/p2_e_s0.full.log 2>&1 || { tail -40 logs/p2_e_s0.full.log; exit 1; }
gzip -f $AP_ROLLOUT_LOG; rm -rf runs/p2_e_s0/checkpoint-*/optimizer.pt
source slurm/p2_evalset.sh dev "E_s0|runs/p2_ewarm/merged|runs/p2_e_s0/final|L" "p1_dev_full p1_dev_rec"
