# minutes: 60
# Amendment 11 secondary (H4/C4): ledger-instructed Aya Expanse 32B judges on the full 815-episode judge pool.
set -uo pipefail
source slurm/claim.sh || exit 0
source slurm/env.sh
.venv/bin/python scripts/freeze_a11.py --check || exit 1
.venv/bin/python scripts/judge_instructed.py --model CohereLabs/aya-expanse-32b --tag aya32b_instructed > logs/507_judge_instructed.full.log 2>&1 \
  || { tail -30 logs/507_judge_instructed.full.log; exit 1; }
tail -40 logs/507_judge_instructed.full.log
