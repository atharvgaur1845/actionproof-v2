# minutes: 30
# SEL1 (PREREGISTRATION_SEL1.md): regenerate the registered splits (deterministic; checked against the registered
# hashes), build the SFT rows of both arms, record their hashes, run the SEL1 tests and the scripted references.
set -uo pipefail
source slurm/claim.sh || exit 0
source slurm/env.sh
mkdir -p logs results/sel1
[ -f data/bench/proc/sel1_test.jsonl ] || .venv/bin/python -m actionproof.sel1.tasks > logs/sel1_splits.log 2>&1 || { tail -20 logs/sel1_splits.log; exit 1; }
.venv/bin/python scripts/freeze_sel1.py --check || exit 1
[ -f data/rft/sel1_sel_rows.jsonl ] || .venv/bin/python -m actionproof.sel1.build > logs/sel1_build.log 2>&1 || { tail -30 logs/sel1_build.log; exit 1; }
tail -45 logs/sel1_build.log
grep -q '"data":' results/sel1_hashes.json || .venv/bin/python scripts/freeze_sel1.py --add-data || exit 1
.venv/bin/python scripts/freeze_sel1.py --check || exit 1
.venv/bin/python -m pytest -q tests/sel1 > logs/sel1_tests.log 2>&1 || { tail -30 logs/sel1_tests.log; exit 1; }
tail -2 logs/sel1_tests.log
.venv/bin/python scripts/sel1_policies_eval.py > logs/sel1_policies.log 2>&1 || { tail -30 logs/sel1_policies.log; exit 1; }
cat logs/sel1_policies.log
echo SEL1_BUILD_OK
