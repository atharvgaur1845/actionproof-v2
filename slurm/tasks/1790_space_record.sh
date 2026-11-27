# minutes: 60
# Live Space fallback: record real runs of the base model and the released agent D (runs/p1_d_s0/merged; its
# model.safetensors sha256 equals the Hub revision 5418df5's, checked below) on every curated task x reality through
# the Space's own episode loop (space_live/record_runs.py). Replayed in the Space only under a RECORDED banner.
set -uo pipefail
source slurm/claim.sh || exit 0
source slurm/env.sh
mkdir -p logs
echo "0e9a257cbc8c56fcc249e6f0f08e1fcbabe3e36f59d32b5fbe420ea8bcd51140  runs/p1_d_s0/merged/model.safetensors" | sha256sum -c - || exit 1
SNAP=$(ls -d $HF_HOME/hub/models--CohereLabs--tiny-aya-global/snapshots/* | head -1)
REV=$(basename $SNAP)
[ -f space_live/data/recorded.json ] || .venv/bin/python space_live/record_runs.py --base $SNAP --base-revision $REV --trained runs/p1_d_s0/merged --trained-revision 5418df563fc3 --hardware "NVIDIA A100 80GB (sharanga), bf16, transformers $(.venv/bin/python -c 'import transformers;print(transformers.__version__)')" --out space_live/data/recorded.json > logs/space_record.log 2>&1 || { tail -30 logs/space_record.log; exit 1; }
tail -52 logs/space_record.log
sha256sum space_live/data/recorded.json; ls -la space_live/data/recorded.json
echo SPACE_RECORD_OK
