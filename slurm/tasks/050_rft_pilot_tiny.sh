# minutes: 60
# R1 pilot: curated + raw RFT data from the 32B teacher pilot, 1-epoch RFT on Tiny Aya,
# then evaluate the adapter on 60 test_iid tasks with the NEUTRAL prompt.
set -euo pipefail
export HF_HUB_OFFLINE=1 VLLM_ENABLE_V1_MULTIPROCESSING=0  # in-process engine (the path TRL colocate uses; subprocess engine core fails to init in this job)
T=results/teacher/pilot_32b_verify_a.jsonl
.venv/bin/python -m actionproof.train.rft_data --teacher $T --out data/rft/pilot_curated.jsonl
.venv/bin/python -m actionproof.train.rft_data --teacher $T --raw --match-curated data/rft/pilot_curated.jsonl --out data/rft/pilot_raw.jsonl
.venv/bin/python -m actionproof.train.rft --data data/rft/pilot_curated.jsonl --out runs/rft_pilot_tiny --epochs 1 2>&1 | grep -E "loss|train_runtime|Error|Traceback" | tail -12
.venv/bin/python -m actionproof.eval.run_eval --model CohereLabs/tiny-aya-global --adapter runs/rft_pilot_tiny/final \
  --split test_iid --n-tasks 60 --prompt neutral --out results/eval/pilot_rft_tiny_neutral.jsonl > "logs/$(basename $0 .sh)_$RANDOM.full.log" 2>&1; tail -25 logs/$(basename $0 .sh)_*.full.log | grep -v -i warn
