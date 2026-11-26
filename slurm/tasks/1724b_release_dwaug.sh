# minutes: 60
# Publish the P4 Dw_aug adapters (seeds 0-2) and refresh every model card (P4 limitation note). Network only.
set -uo pipefail
source slurm/claim.sh || exit 0
source slurm/env.sh
unset HF_HUB_OFFLINE
for s in 0 1 2; do [ -f runs/p4_dwaug_s$s/final/adapter_config.json ] || { echo "missing runs/p4_dwaug_s$s/final"; exit 1; }; done
.venv/bin/python scripts/release_hf.py --execute --only dwaug merged-d d-adapters dw a dn > logs/release_dwaug.log 2>&1 || { tail -40 logs/release_dwaug.log; exit 1; }
tail -8 logs/release_dwaug.log
echo RELEASE_DWAUG_OK
