# minutes: 150
# H7: knowing vs ignorant liar across checkpoints of the seed-0 arms (every 50 steps + final).
set -uo pipefail
source slurm/env.sh
for arm in tool_trust pcstr outcome; do
  [ -d runs/grpo_tiny_${arm}_s0 ] || continue
  .venv/bin/python -m actionproof.interp.knowing_liar --run runs/grpo_tiny_${arm}_s0 --arm $arm \
    --probe results/interp/probe_cues_B.npz --out results/interp/kl_${arm}.json 2>&1 | grep -v -i warn | tail -8
done
