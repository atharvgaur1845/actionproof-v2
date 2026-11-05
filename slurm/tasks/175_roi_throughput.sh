# minutes: 20
# after: 150_grpo_v2_pcstr_d
# ROI: time a batched vLLM run of the final PCSTR-d agent on test_iid (200 tickets) on this GPU.
set -euo pipefail
source slurm/env.sh
mkdir -p results/roi
O=results/roi/throughput_episodes.jsonl
t0=$(date +%s)
.venv/bin/python -m actionproof.eval.run_eval --model runs/rft_tiny_v2/merged --adapter runs/grpo_tiny_v2_pcstr_d_s0/final \
  --split test_iid --prompt neutral --out $O > logs/175_roi.full.log 2>&1
t1=$(date +%s)
gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
python3 -c "import json;s=$t1-$t0;json.dump({'gpu':'$gpu','episodes':200,'seconds':s,'episodes_per_hour':200*3600/s,'note':'wall time incl. vLLM start-up and merge (conservative)'},open('results/roi/throughput.json','w'));print(open('results/roi/throughput.json').read())"
.venv/bin/python -m actionproof.eval.roi --episodes $O --throughput results/roi/throughput.json --prices configs/roi_prices.json
