# minutes: 15
# Amendment 4 S0/S5 integrity on the cluster: hashes, exclusion audit (real prompt rendering + init lineage), tests.
set -euo pipefail
source slurm/env.sh
.venv/bin/python scripts/freeze_bench.py --check
.venv/bin/python scripts/freeze_a4.py --check
.venv/bin/python scripts/a4_exclusion_audit.py --init-runs runs/rft_tiny_v2 | tail -12
.venv/bin/python -m pytest -q tests/ 2>&1 | tail -2
