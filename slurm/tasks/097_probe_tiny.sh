# minutes: 45
# R2: activations of the RFT init on its own curriculum samples (train split, T=1.0, 8/task),
# per-layer linear probes at pos-A (acked write did not land) and pos-B (task not done at report),
# F1 negative control, text baseline, and layer-28 SAE features (FVU go/no-go).
set -euo pipefail
source slurm/env.sh
mkdir -p data/interp results/interp
[ -f data/interp/rft_tiny_train.npz ] || .venv/bin/python -m actionproof.interp.extract \
  --episodes results/curriculum/rft_tiny_train.jsonl --model runs/rft_tiny_curated/merged \
  --out data/interp/rft_tiny_train.npz > logs/097_extract.full.log 2>&1
tail -1 logs/097_extract.full.log
[ -f results/interp/rft_probe_cues.jsonl ] || .venv/bin/python -m actionproof.eval.run_eval --model CohereLabs/tiny-aya-global \
  --adapter runs/rft_tiny_curated/final --split probe_cues --prompt neutral --samples 4 --temperature 1.0 \
  --out results/interp/rft_probe_cues.jsonl > logs/097_cues_eval.full.log 2>&1
[ -f data/interp/rft_probe_cues.npz ] || .venv/bin/python -m actionproof.interp.extract \
  --episodes results/interp/rft_probe_cues.jsonl --model runs/rft_tiny_curated/merged \
  --out data/interp/rft_probe_cues.npz > logs/097_extract_cues.full.log 2>&1
tail -1 logs/097_extract_cues.full.log
for p in A B; do
  .venv/bin/python -m actionproof.interp.probe --data data/interp/rft_probe_cues.npz --pos $p \
    --out results/interp/probe_cues_$p.json --save-probe results/interp/probe_cues_$p.npz
done
for p in A B; do
  .venv/bin/python -m actionproof.interp.probe --data data/interp/rft_tiny_train.npz --pos $p \
    --out results/interp/probe_$p.json --save-probe results/interp/probe_$p.npz
  .venv/bin/python -m actionproof.interp.sae_features --data data/interp/rft_tiny_train.npz --pos $p \
    --out results/interp/sae_$p.json
done
