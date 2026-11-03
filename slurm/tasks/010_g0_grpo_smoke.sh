# minutes: 60
# G0 (c)+(d): 2-step GRPO on Tiny Aya with per-rollout envs; asserts rollout-id <-> receipt link,
# measures step time. Pass = finishes, reward keys reach reward_funcs, checkpoint written.
set -euo pipefail
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
.venv/bin/python -m pytest -q tests 2>&1 | tail -2
time .venv/bin/python -m actionproof.train.grpo --arm pcstr --model CohereLabs/tiny-aya-global \
  --out runs/g0_grpo_smoke --n-tasks 8 --steps 2 --num-generations 4 --prompts-per-step 2 --save-steps 2 \
  2>&1 | grep -v -E "Warning|warn\(" | tail -60
.venv/bin/python - <<'PY'
import json; h=json.load(open("runs/g0_grpo_smoke/log_history.json"))
for r in h: print({k: r[k] for k in r if any(s in k for s in ("reward", "step_time", "loss", "step", "runtime"))})
PY
ls runs/g0_grpo_smoke
