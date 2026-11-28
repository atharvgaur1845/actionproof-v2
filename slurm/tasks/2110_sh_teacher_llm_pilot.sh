# minutes: 150
# Self-heal, PREPARED, NOT SUBMITTED: LLM teacher pilot. Serves CohereLabs/aya-expanse-32b with vLLM on this GPU and
# runs the correction step with --teacher llm on the same DEV failures as the B4 pilot (r0_dev). Every proposed
# continuation is replayed and kept only if the verifier passes it. No API spend: the model is local.
set -uo pipefail
source wt/slurm/claim.sh || exit 0
source wt/slurm/selfheal/round.sh
R=$AP/results
PORT=18${SLURM_JOB_ID: -3}
$AP/.venv/bin/vllm serve CohereLabs/aya-expanse-32b --port $PORT --max-model-len 8192 --gpu-memory-utilization 0.9 \
  --seed 0 > logs/llm_teacher_serve.log 2>&1 &
SRV=$!; trap 'kill $SRV 2>/dev/null' EXIT
for i in $(seq 1 120); do curl -sf localhost:$PORT/v1/models >/dev/null && break; sleep 10; done
O=$SH/results/r0_dev_llm; mkdir -p $O
$PY scripts/selfheal_run.py correct --teacher llm --api-base http://localhost:$PORT/v1 --api-model CohereLabs/aya-expanse-32b \
  --out $O/corrections_p1_dev.jsonl --inputs $R/p1/dev/D_s{0,1,2}__p1_dev_full.jsonl $R/p1/dev/D_s{0,1,2}__p1_dev_rec.jsonl \
  > logs/llm_teacher_p1.log 2>&1 || { tail -30 logs/llm_teacher_p1.log; exit 1; }
$PY scripts/selfheal_run.py correct --teacher llm --api-base http://localhost:$PORT/v1 --api-model CohereLabs/aya-expanse-32b \
  --out $O/corrections_txn_dev.jsonl --inputs $R/rg1/dev_D_s0_none__txn_dev.jsonl > logs/llm_teacher_txn.log 2>&1 \
  || { tail -30 logs/llm_teacher_txn.log; exit 1; }
cat $O/*.stats.json
echo SH_LLM_TEACHER_OK
