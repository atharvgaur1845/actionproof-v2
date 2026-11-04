# minutes: 200
# after: 091_curriculum_v2
# R3 round v2 / H2: GRPO seed 0, arm=llm_judge (Aya Expanse 8B transcript judge, likelihood of COMPLETE; H2 + D1 link), shared RFT v2 init.
set -euo pipefail
source slurm/env.sh
slurm/grpo_arm.sh llm_judge runs/rft_tiny_v2 data/curriculum/band_tiny_v2.jsonl v2
