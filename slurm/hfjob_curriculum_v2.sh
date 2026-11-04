#!/bin/bash
# HF Jobs re-run of queue task 091 (v2 learnable band) after job 6aeb691e0e0338d1bb800c4e was cancelled
# on HF's side mid-curriculum; re-run 2026-11-05 after the sampling-seed fix (Amendment 3). Same env (bundle), RFT v2 adapter from outputs/, latest code overlay.
set -euo pipefail
W=/work; mkdir -p $W && cd $W
export HF_HOME=$W/.cache/huggingface UV_CACHE_DIR=$W/.cache/uv TOKENIZERS_PARALLELISM=false
export TMPDIR=$W/.cache/tmp VLLM_CACHE_ROOT=$W/.cache/vllm TRITON_CACHE_DIR=$W/.cache/triton
export VLLM_USE_FLASHINFER_SAMPLER=0 VLLM_ENABLE_V1_MULTIPROCESSING=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p $TMPDIR
uvx hf download "$AP_REPO" bundle/actionproof.tar.gz bundle/code.tar.gz --repo-type dataset --local-dir /tmp/b
tar xzf /tmp/b/bundle/actionproof.tar.gz -C $W && tar xzf /tmp/b/bundle/code.tar.gz -C $W
uvx hf download "$AP_REPO" --repo-type dataset --include "outputs/runs/rft_tiny_v2/final/*" --local-dir /tmp/o
mkdir -p runs/rft_tiny_v2 && cp -r /tmp/o/outputs/runs/rft_tiny_v2/final runs/rft_tiny_v2/
uv venv -q --python 3.12 .venv
grep -v '^-e ' requirements.lock > /tmp/req.txt
uv pip install -q --python .venv -r /tmp/req.txt
uv pip install -q --python .venv --no-deps -e vendor/AgentTune -e vendor/AuditKIT -e vendor/CuratorKIT -e .
mkdir -p logs results/curriculum data/curriculum
S=results/curriculum/rft_tiny_v2_train.jsonl
date +%T; echo "curriculum start"
.venv/bin/python -m actionproof.eval.run_eval --model CohereLabs/tiny-aya-global --adapter runs/rft_tiny_v2/final \
  --split train --prompt neutral --samples 8 --temperature 1.0 --out $S > logs/091_curriculum.full.log 2>&1 \
  || { tail -40 logs/091_curriculum.full.log; exit 1; }
date +%T; echo "curriculum sampled"
.venv/bin/python -m actionproof.train.curriculum --samples $S --out data/curriculum/band_tiny_v2.jsonl
.venv/bin/python -m actionproof.eval.by_fault $S
V=results/eval/r1/rft_tiny_v2__neutral__val_T1x4
.venv/bin/python -m actionproof.eval.run_eval --model CohereLabs/tiny-aya-global --adapter runs/rft_tiny_v2/final \
  --split val --prompt neutral --samples 4 --temperature 1.0 --out $V.jsonl > logs/091_val_T1.full.log 2>&1 || tail -20 logs/091_val_T1.full.log
.venv/bin/python -m actionproof.eval.by_fault $V.jsonl || true
mkdir -p results/eval/r1
for p in results/curriculum data/curriculum results/eval/r1 logs; do .venv/bin/hf upload "$AP_REPO" $p outputs/$p --repo-type dataset --quiet; done
echo HFJOB_DONE
