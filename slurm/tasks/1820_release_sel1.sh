# minutes: 90
# Publish the SEL1 adapters (sel/s0-s2, ctl/s0-s2) with their card, and add results/sel1 + the SEL1 splits to the
# traces dataset (refreshed dataset card). After the registered analysis (task 1812). Network only.
set -uo pipefail
source slurm/claim.sh || exit 0
source slurm/env.sh
unset HF_HUB_OFFLINE
for a in SEL CTL; do for s in 0 1 2; do [ -f runs/sel1_${a}_s$s/final/adapter_config.json ] || { echo "missing runs/sel1_${a}_s$s/final"; exit 1; }; done; done
.venv/bin/python scripts/freeze_sel1.py --check || exit 1
.venv/bin/python scripts/release_hf.py --execute --only sel1 dataset > logs/release_sel1.log 2>&1 || { tail -40 logs/release_sel1.log; exit 1; }
tail -12 logs/release_sel1.log
echo RELEASE_SEL1_OK
