# minutes: 200
# R3 round v2: GRPO seed 0, arm=tool_trust, from the shared RFT v2 init (see 087), v2 learnable band.
set -euo pipefail
source slurm/env.sh
slurm/grpo_arm.sh tool_trust runs/rft_tiny_v2 data/curriculum/band_tiny_v2.jsonl v2
