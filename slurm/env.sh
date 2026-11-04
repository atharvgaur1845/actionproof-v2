# Sourced by every queue task. Tasks may run inside another project's job (train hook), so
# override every inherited cache path: never write into /scratch/pabitra/gemma4comp.
R=/scratch/pabitra/actionproof
export HF_HOME=/scratch/pabitra/.cache/huggingface HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export TMPDIR=$R/.cache/tmp VLLM_CACHE_ROOT=$R/.cache/vllm TRITON_CACHE_DIR=$R/.cache/triton
export FLASHINFER_WORKSPACE_BASE=$R/.cache/flashinfer OUTLINES_CACHE_DIR=$R/.cache/outlines
mkdir -p "$TMPDIR" "$VLLM_CACHE_ROOT" "$TRITON_CACHE_DIR" "$FLASHINFER_WORKSPACE_BASE" "$OUTLINES_CACHE_DIR"
unset CUDA_HOME LIBRARY_PATH LD_LIBRARY_PATH VLLM_ALLOW_RUNTIME_LORA_UPDATING
# FlashInfer's top-k/top-p sampler JIT-compiles and fails to link against pip CUDA wheels
# (-lcudart). vLLM's PyTorch sampler is used instead; no JIT needed.
export VLLM_USE_FLASHINFER_SAMPLER=0 VLLM_ENABLE_V1_MULTIPROCESSING=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# Drop PATH entries that are missing or not searchable on this node (gpunode7: /apps/ngc). Otherwise any
# PATH lookup of an absent tool raises PermissionError instead of FileNotFoundError: torch inductor's
# `nvcc --version` probe (used while writing compile repro info) then crashes vLLM's CUDA-graph compile.
_p=""; for _d in ${PATH//:/ }; do [ -d "$_d" ] && [ -x "$_d" ] && _p="${_p:+$_p:}$_d"; done; export PATH="$_p"; unset _p _d
