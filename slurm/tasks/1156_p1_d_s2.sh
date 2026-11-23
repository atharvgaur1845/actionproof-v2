# minutes: 210
# P1 practical agent (PREREGISTRATION_P1.md): arm D seed 2: RSC recipe (R_V, s3_trainb_half + s3_rec_train_half, 200 steps) from the recovery-SFT warm start
# after: 1153_p1_dwarm
set -uo pipefail
source slurm/claim.sh || exit 0
source slurm/env.sh
.venv/bin/python scripts/freeze_p1.py --check || exit 1
export MASTER_PORT=29952 AP_ROLLOUT_LOG=results/p1/rollouts/D_s2.jsonl
mkdir -p results/p1/rollouts
[ -f runs/p1_d_s2/final/adapter_config.json ] || .venv/bin/python -m actionproof.stage3.train --reward rv \
  --pools data/bench/proc/s3_trainb_half.jsonl,data/bench/proc/s3_rec_train_half.jsonl --seed 2 --steps 200 \
  --init-adapter runs/p1_dwarm/final --out runs/p1_d_s2 --use-vllm > logs/p1_d_s2.full.log 2>&1 || { tail -40 logs/p1_d_s2.full.log; exit 1; }
gzip -f $AP_ROLLOUT_LOG; rm -rf runs/p1_d_s2/checkpoint-*/optimizer.pt
source slurm/p1_evalset.sh dev "D_s2|runs/p1_dwarm/merged|runs/p1_d_s2/final|0" "p1_dev_full p1_dev_rec"
