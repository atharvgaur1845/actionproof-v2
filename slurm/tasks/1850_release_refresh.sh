# minutes: 90
# Refresh every model card and the dataset card after the final hostile review (wording; no model file changes), and
# add the SEL1 training rows (data/rft/sel1_*) to the traces dataset. Network only.
set -uo pipefail
source slurm/claim.sh || exit 0
source slurm/env.sh
unset HF_HUB_OFFLINE
.venv/bin/python scripts/freeze_sel1.py --check || exit 1
.venv/bin/python scripts/release_hf.py --execute --only merged-d d-adapters dw a dn dwaug sel1 dataset > logs/release_refresh.log 2>&1 || { tail -40 logs/release_refresh.log; exit 1; }
grep -i -E "upload|skip|commit" logs/release_refresh.log | tail -20
echo RELEASE_REFRESH_OK
