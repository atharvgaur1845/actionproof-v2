# minutes: 120
# Amendment 11 (PREREGISTRATION_A11, H1): PCSTR-d seed 1 x {passthrough, enforce, enforce_inform} x {test_iid_c2, v2_stress_c2}, greedy, 12 turns, neutral prompt.
# Not executed when written (2026-11-20); runs only after scripts/freeze_a11.py has registered A11.
# Dual-pool copy (a100 + h200): run once only. If the other pool already claimed or finished it, skip.
for f in /scratch/pabitra/actionproof/queue/*/running/$(basename "$0") /scratch/pabitra/actionproof/queue/*/done/$(basename "$0"); do
  [ -e "$f" ] && [ "$(readlink -f "$f")" != "$(readlink -f "$0")" ] && { echo "SKIP: $f"; exit 0; }
done
set -uo pipefail
source slurm/env.sh
.venv/bin/python scripts/freeze_a11.py --check || exit 1
slurm/a11_policy.sh pcstr_d_s1 runs/rft_tiny_v2/merged "runs/grpo_tiny_v2_pcstr_d_s1/final" ""
