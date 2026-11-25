# minutes: 240
# Judge reliability study (research_audit/JUDGE_RELIABILITY.md): build the dataset, then run Aya Expanse 8B and 32B
# on seven judge configurations (evaluation/judge_next/run.py). Local inference, no API spend.
set -uo pipefail
source slurm/claim.sh || exit 0
source slurm/env.sh
mkdir -p logs results/judge_next
.venv/bin/python evaluation/judge_next/build.py > logs/judge_next_build.log 2>&1 || { cat logs/judge_next_build.log; exit 1; }
cat logs/judge_next_build.log
.venv/bin/python evaluation/judge_next/run.py --model CohereLabs/aya-expanse-8b --tag aya8b > logs/judge_next_aya8b.log 2>&1 || { tail -30 logs/judge_next_aya8b.log; exit 1; }
.venv/bin/python evaluation/judge_next/run.py --model CohereLabs/aya-expanse-32b --tag aya32b > logs/judge_next_aya32b.log 2>&1 || { tail -30 logs/judge_next_aya32b.log; exit 1; }
echo JUDGE_NEXT_OK
