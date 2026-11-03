# minutes: 120
# R1 pilot: Aya Expanse 32B teacher with verify_a, 50 train tasks x 4 samples (T=0.7).
# Measures teacher VSC (RFT keep-rate) and throughput before the full 400x8 run.
set -euo pipefail
export HF_HUB_OFFLINE=1 VLLM_ENABLE_V1_MULTIPROCESSING=0  # in-process engine (the path TRL colocate uses; subprocess engine core fails to init in this job)
.venv/bin/python -m actionproof.eval.run_eval --model CohereLabs/aya-expanse-32b --split train \
  --n-tasks 50 --prompt verify_a --samples 4 --temperature 0.7 --gpu-mem 0.88 \
  --out results/teacher/pilot_32b_verify_a.jsonl > "logs/$(basename $0 .sh)_$RANDOM.full.log" 2>&1; tail -25 logs/$(basename $0 .sh)_*.full.log | grep -v -i warn
