# minutes: 240
# TXN1 supplementary API baseline (PREREGISTRATION_TXN1.md): Qwen3-235B-A22B-Instruct-2507 via the HF router, greedy,
# hard stop at $3 (same router-listed prices as scripts/b1_api_eval.py: $0.09 in / $0.55 out per 1M). Network only.
set -uo pipefail
source slurm/claim.sh || exit 0
source slurm/env.sh
.venv/bin/python scripts/freeze_txn1.py --check || exit 1
[ -f results/txn/qwen__txn_test.summary.json ] && { echo exists; exit 0; }
export OPENAI_API_KEY="$(cat ${HF_TOKEN_PATH:-/scratch/pabitra/.cache/huggingface/token})"
.venv/bin/python scripts/txn_eval.py --tag qwen --split txn_test --api-base https://router.huggingface.co/v1 \
  --api-model Qwen/Qwen3-235B-A22B-Instruct-2507:deepinfra --price-in 0.09 --price-out 0.55 --max-usd 3 \
  --concurrency 8 || exit 1
echo TXN_QWEN_OK
