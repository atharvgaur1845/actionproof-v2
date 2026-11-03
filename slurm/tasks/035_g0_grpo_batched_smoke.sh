# minutes: 60
# G0 (d) re-test: batched lockstep rollouts on colocated vLLM, real step size (4 prompts x 8).
# Pass = 3 steps complete, reward variance > 0, step_time well under 180 s.
set -euo pipefail
source slurm/env.sh
rm -rf runs/g0_grpo_batched
time .venv/bin/python -m actionproof.train.grpo --arm pcstr --model CohereLabs/tiny-aya-global \
  --out runs/g0_grpo_batched --n-tasks 32 --steps 3 --num-generations 8 --prompts-per-step 4 --save-steps 3 \
  --use-vllm --rollout batched 2>&1 | grep -v -E "Warning|warn\(" | tail -40
.venv/bin/python - <<'PY'
import json; h=json.load(open("runs/g0_grpo_batched/log_history.json"))
for r in h: print({k: r[k] for k in r if k in ("step","reward","reward_std","frac_reward_zero_std","step_time","loss","kl","completions/mean_length")})
PY
