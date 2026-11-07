# minutes: 200
# after: 332_a4_gate
# Amendment 4 X3 / H10: ack_outcome GRPO, seed 0. IDENTICAL config to the v2 arms (grpo_arm.sh, v2 band, shared init;
# dev eval with grpo_arm.sh's default turn budget, as for the outcome arm it is compared with; see D-A4-1).
set -euo pipefail
source slurm/env.sh
bash slurm/grpo_arm.sh ack_outcome runs/rft_tiny_v2 data/curriculum/band_tiny_v2.jsonl v2 0
.venv/bin/python scripts/a4_log_run.py --id X3_ack_outcome_s0 --kind grpo --seed 0 --cmd "slurm/grpo_arm.sh ack_outcome runs/rft_tiny_v2 data/curriculum/band_tiny_v2.jsonl v2 0" --adapter runs/grpo_tiny_v2_ack_outcome_s0/final --data data/curriculum/band_tiny_v2.jsonl > /dev/null
