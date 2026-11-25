# minutes: 90
# Public HF release (author-approved 2026-11-25): merged D s0, adapters (D, Dw, A+W, Dn), cards; then the traces dataset.
set -uo pipefail
source slurm/claim.sh || exit 0
source slurm/env.sh
unset HF_HUB_OFFLINE
.venv/bin/python scripts/release_hf.py --execute --only merged-d d-adapters dw a dn > logs/release_models.log 2>&1 || { tail -40 logs/release_models.log; exit 1; }
tail -5 logs/release_models.log
.venv/bin/python scripts/release_hf.py --execute --only dataset > logs/release_dataset.log 2>&1 || { tail -40 logs/release_dataset.log; exit 1; }
tail -5 logs/release_dataset.log
echo RELEASE_OK
