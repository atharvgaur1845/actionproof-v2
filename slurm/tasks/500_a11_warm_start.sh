# minutes: 120
# Amendment 11 (PREREGISTRATION_A11, H1): warm start rft_tiny_v2 (no GRPO) x {passthrough, enforce, enforce_inform} x {test_iid_c2, v2_stress_c2}, greedy, 12 turns, neutral prompt.
# Not executed when written (2026-11-20); runs only after scripts/freeze_a11.py has registered A11.
set -uo pipefail
source slurm/claim.sh || exit 0
source slurm/env.sh
.venv/bin/python scripts/freeze_a11.py --check || exit 1
slurm/a11_policy.sh warm_start runs/rft_tiny_v2/merged "" ""
