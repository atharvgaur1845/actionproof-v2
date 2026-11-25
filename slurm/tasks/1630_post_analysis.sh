# minutes: 30
# CPU analysis after P3/judge_next/Dw: exploratory 2x2 (scripts/audit/dwarm_2x2.py) and judge_next metrics (evaluation/judge_next/analyze.py).
set -uo pipefail
source slurm/claim.sh || exit 0
source slurm/env.sh
.venv/bin/python scripts/audit/dwarm_2x2.py || exit 1
.venv/bin/python evaluation/judge_next/analyze.py || exit 1
echo POST_ANALYSIS_OK
