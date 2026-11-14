#!/bin/bash
# Amendment 8 addendum 2 (PREREGISTRATION_A8_ADDENDUM2): second seed of the A7 acknowledgement arm (task 412 at seed 1).
# Identical to 412 except --seed 1; training rollouts logged. Run through the existing A7 arm script.
set -euo pipefail
.venv/bin/python scripts/freeze_a8c.py --check
export AP_ROLLOUT_LOG=runs/grpo_tiny_a7_ack_outcome_s1/train_rollouts.jsonl
bash slurm/a7_arm.sh ack_outcome 1 rft_tiny_a7 a7
