# R1-T τ-bench retail (PREREGISTRATION_R1_TAU.md): one H200 holds the simulated user (Aya Expanse 32B) and the agent.
#   usage (from a task script): source slurm/r1_tau_set.sh "tag=run_dir ..."   (R1T_LIMIT / R1T_TRIALS / R1T_OUT optional)
.venv/bin/python scripts/freeze_r1_tau_d2.py --check || exit 1   # D-R1T-1 (user context 8192) + D-R1T-2 (claim-safe cleanup)
mkdir -p logs results/r1_tau
P=$((8100 + ${LANE:-0} * 10))
export AP_TAU_USER_BASE=http://127.0.0.1:$P/v1 AP_TAU_AGENT_BASE=http://127.0.0.1:$((P+1))/v1 AP_TAU_MAX_TOKENS=512
export PYTHONPATH=$PWD/actionproof/retention/tau_shim:/scratch/pabitra/ext/tau-bench:$PWD
wait_up() { for _ in $(seq 120); do curl -sf "http://127.0.0.1:$1/health" >/dev/null && return 0; sleep 10; done; return 1; }
.venv/bin/vllm serve CohereLabs/aya-expanse-32b --served-model-name user --port $P --max-model-len 8192 \
  --gpu-memory-utilization 0.62 --generation-config vllm --seed 0 > logs/r1t_user_$P.log 2>&1 &
UPID=$!
# D-R1T-2: no EXIT trap here (it replaced slurm/claim.sh's claim-release trap); servers are stopped on every exit path
die() { kill $UPID ${APID:-} 2>/dev/null; exit 1; }
wait_up $P || { tail -30 logs/r1t_user_$P.log; die; }
for m in $1; do
  tag=${m%%=*}; run=${m#*=}; made=""
  case $tag in
    base) model=CohereLabs/tiny-aya-global ;;
    warm_start) model=runs/rft_proc/merged ;;
    *) model=$run/merged
       [ -f $model/config.json ] || { .venv/bin/python -m actionproof.train.merge $run/final runs/rft_proc/merged \
         > logs/r1t_merge_$tag.log 2>&1 || die; made=1; } ;;
  esac
  .venv/bin/vllm serve $model --served-model-name agent --port $((P+1)) --max-model-len 8192 \
    --gpu-memory-utilization 0.15 --generation-config vllm --seed 0 > logs/r1t_agent_$tag.log 2>&1 &
  APID=$!
  wait_up $((P+1)) || { tail -30 logs/r1t_agent_$tag.log; die; }
  .venv/bin/python -m actionproof.retention.tau_run --tag $tag --trials ${R1T_TRIALS:-3} ${R1T_LIMIT:+--limit $R1T_LIMIT} \
    ${R1T_OUT:+--out-dir $R1T_OUT} > logs/r1t_run_$tag.log 2>&1 || { tail -30 logs/r1t_run_$tag.log; die; }
  kill $APID; wait $APID 2>/dev/null; APID=""
  [ -n "$made" ] && rm -rf $model
  echo "$(date +%T) done $tag: $(cat ${R1T_OUT:-results/r1_tau}/$tag.summary.json | tr -d '\n ')"
done
kill $UPID 2>/dev/null; wait $UPID 2>/dev/null
