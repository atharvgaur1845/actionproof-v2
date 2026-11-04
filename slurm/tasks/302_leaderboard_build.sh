# minutes: 10
# after: 300_leaderboard_small 301_leaderboard_32b
# Aggregate all leaderboard cells (Wilson CIs) into results/leaderboard.{json,md}.
set -euo pipefail
source slurm/env.sh
.venv/bin/python -m actionproof.eval.leaderboard --glob 'results/eval/lb/*.jsonl' --out results/leaderboard
