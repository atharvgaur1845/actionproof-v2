# minutes: 90
# Refresh the traces dataset with the final P4, P5, B1 and TXN results. Network only.
set -uo pipefail
source slurm/claim.sh || exit 0
source slurm/env.sh
unset HF_HUB_OFFLINE
.venv/bin/python scripts/release_hf.py --execute --only dataset > logs/release_dataset2.log 2>&1 || { tail -40 logs/release_dataset2.log; exit 1; }
tail -5 logs/release_dataset2.log
echo RELEASE_DATASET_OK
