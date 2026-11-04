#!/bin/bash
# HF Jobs run of queue tasks 087 (RFT v2: resume + val eval) and 091 (v2 curriculum), used while the
# cluster had no free GPU (2026-11-04). Same commands, same pinned environment (requirements.lock),
# same seeds. Inputs and outputs live in the private repo $AP_REPO (dataset type).
#   hf jobs run --flavor h200 --timeout 3h --secrets HF_TOKEN -e AP_REPO=... ghcr.io/astral-sh/uv:python3.12-bookworm \
#       bash -c "uvx hf download \$AP_REPO bundle/hfjob_rft_v2.sh --repo-type dataset --local-dir /tmp/b && bash /tmp/b/bundle/hfjob_rft_v2.sh"
set -euo pipefail
W=/work; mkdir -p $W && cd $W
export HF_HOME=$W/.cache/huggingface UV_CACHE_DIR=$W/.cache/uv TOKENIZERS_PARALLELISM=false
export TMPDIR=$W/.cache/tmp VLLM_CACHE_ROOT=$W/.cache/vllm TRITON_CACHE_DIR=$W/.cache/triton
export VLLM_USE_FLASHINFER_SAMPLER=0 VLLM_ENABLE_V1_MULTIPROCESSING=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p $TMPDIR
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader

uvx hf download "$AP_REPO" bundle/actionproof.tar.gz --repo-type dataset --local-dir /tmp/b
tar xzf /tmp/b/bundle/actionproof.tar.gz -C $W
uv venv -q --python 3.12 .venv
grep -v '^-e ' requirements.lock > /tmp/req.txt
uv pip install -q --python .venv -r /tmp/req.txt
uv pip install -q --python .venv --no-deps -e vendor/AgentTune -e vendor/AuditKIT -e vendor/CuratorKIT -e .
.venv/bin/python -c "import torch, vllm, transformers, trl, agenttune; print('torch', torch.__version__, torch.version.cuda, torch.cuda.is_available(), 'vllm', vllm.__version__)"

upload() {  # push outputs as they appear, so a timeout never loses finished work
  .venv/bin/hf upload "$AP_REPO" "$1" "outputs/$1" --repo-type dataset --quiet || true
}
mkdir -p logs results/eval/r1 results/curriculum data/curriculum

# --- 087: RFT v2 (resumes from runs/rft_tiny_v2/checkpoint-*) + val eval
.venv/bin/python -m actionproof.train.rft --data data/rft/curated_v2_32b.jsonl --out runs/rft_tiny_v2 --epochs 2 \
  > logs/087_rft_v2.full.log 2>&1 || { tail -40 logs/087_rft_v2.full.log; upload logs; exit 1; }
grep -E "train_runtime" logs/087_rft_v2.full.log | tail -1 | cut -c1-200
rm -rf runs/rft_tiny_v2/checkpoint-*/optimizer.pt
upload runs/rft_tiny_v2/final
R=results/eval/r1/rft_tiny_v2__neutral__val
.venv/bin/python -m actionproof.eval.run_eval --model CohereLabs/tiny-aya-global --adapter runs/rft_tiny_v2/final \
  --split val --prompt neutral --out $R.jsonl > logs/087_eval_v2.full.log 2>&1 || { tail -40 logs/087_eval_v2.full.log; upload logs; exit 1; }
.venv/bin/python -m actionproof.eval.by_fault $R.jsonl
upload results/eval/r1

# --- 091: learnable band for the v2 init (8 samples per train task, T=1.0)
S=results/curriculum/rft_tiny_v2_train.jsonl
.venv/bin/python -m actionproof.eval.run_eval --model CohereLabs/tiny-aya-global --adapter runs/rft_tiny_v2/final \
  --split train --prompt neutral --samples 8 --temperature 1.0 --out $S > logs/091_curriculum.full.log 2>&1 \
  || { tail -40 logs/091_curriculum.full.log; upload logs; exit 1; }
.venv/bin/python -m actionproof.train.curriculum --samples $S --out data/curriculum/band_tiny_v2.jsonl
.venv/bin/python -m actionproof.eval.by_fault $S
upload results/curriculum
upload data/curriculum
upload logs
echo HFJOB_DONE
