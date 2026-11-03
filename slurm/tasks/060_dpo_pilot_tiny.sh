# minutes: 60
# R1 pilot: first-divergence DPO pairs from teacher pilot samples, 1-epoch DPO on top of the
# RFT pilot adapter, then evaluate with the NEUTRAL prompt on the same 60 test_iid tasks.
set -euo pipefail
source slurm/env.sh
.venv/bin/python -m actionproof.train.dpo_data --samples results/teacher/pilot_32b_verify_a.jsonl results/eval/pilot_rft_tiny_neutral.jsonl --out data/dpo/pilot_pairs.jsonl
.venv/bin/python -m actionproof.train.dpo --pairs data/dpo/pilot_pairs.jsonl --init-adapter runs/rft_pilot_tiny/final --out runs/dpo_pilot_tiny --epochs 1 2>&1 | grep -E "loss|rewards/|train_runtime|Error|Traceback" | tail -12
