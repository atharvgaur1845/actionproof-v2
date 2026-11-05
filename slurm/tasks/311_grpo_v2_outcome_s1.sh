# minutes: 160
# after: 091_curriculum_v2
# R3 round v2, seed 1 (statistics: hierarchical bootstrap over seeds): arm=outcome, shared RFT v2 init.
set -euo pipefail
source slurm/env.sh
slurm/grpo_arm.sh outcome runs/rft_tiny_v2 data/curriculum/band_tiny_v2.jsonl v2 1
