# minutes: 30
# after: 1850_release_refresh
# Re-upload the model cards after a last wording change (D card scope line). Unchanged files are skipped. Network only.
set -uo pipefail
source slurm/claim.sh || exit 0
source slurm/env.sh
unset HF_HUB_OFFLINE
.venv/bin/python scripts/release_hf.py --execute --only merged-d d-adapters dw a dn dwaug sel1 > logs/cards_refresh.log 2>&1 || { tail -30 logs/cards_refresh.log; exit 1; }
tail -6 logs/cards_refresh.log
echo CARDS_REFRESH_OK
