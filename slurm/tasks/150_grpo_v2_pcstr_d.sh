# minutes: 200
# after: 091_curriculum_v2
# R3 round v2: GRPO seed 0, arm=pcstr_d, from the shared RFT v2 init (see 087), v2 learnable band.
set -euo pipefail
source slurm/env.sh
slurm/grpo_arm.sh pcstr_d runs/rft_tiny_v2 data/curriculum/band_tiny_v2.jsonl v2
