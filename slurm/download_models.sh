#!/usr/bin/env bash
# Pre-download all permitted models + Tiny Aya SAEs to /scratch (compute nodes may lack internet).
set -uo pipefail
export HF_HOME=/scratch/$USER/.cache/huggingface HF_HUB_ENABLE_HF_TRANSFER=0
PY=/scratch/$USER/actionproof/.venv/bin/python
for repo in CohereLabs/tiny-aya-global Farseen0/tiny-aya-saes CohereLabs/aya-expanse-8b \
            CohereLabs/aya-expanse-32b CohereLabs/tiny-aya-fire CohereLabs/tiny-aya-water \
            CohereLabs/tiny-aya-earth CohereLabs/tiny-aya-base; do
  echo "== $(date +%T) $repo"
  $PY -c "from huggingface_hub import snapshot_download as s; print(s('$repo', allow_patterns=['*.json','*.safetensors','*.model','*.txt','*.jinja','*.md']))" \
    || echo "FAILED $repo"
done
echo DOWNLOADS_DONE
