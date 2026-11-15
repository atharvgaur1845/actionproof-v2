# minutes: 330
# Amendment 9 (PREREGISTRATION_A9 §3): the round-v2 llm_judge arm with the receipt ledger in the judge prompt, seed 0.
# Dual-pool copy (a100 + h200): run once only. If the other pool already claimed or finished it, skip.
for f in /scratch/pabitra/actionproof/queue/*/running/421_*.sh /scratch/pabitra/actionproof/queue/*/done/421_*.sh; do
  [ -e "$f" ] && [ "$(readlink -f "$f")" != "$(readlink -f "$0")" ] && { echo "SKIP: $f"; exit 0; }
done
set -euo pipefail
source slurm/env.sh
.venv/bin/python scripts/freeze_a9.py --check
slurm/grpo_arm.sh grounded_judge runs/rft_tiny_v2 data/curriculum/band_tiny_v2.jsonl v2
