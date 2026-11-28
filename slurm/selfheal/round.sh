# Self-heal loop round library (docs/selfheal/DESIGN.md). Sourced by slurm/tasks/21xx_sh_*.sh, which the self-heal
# worker (slurm/selfheal/worker.sh, ROOT=/scratch/pabitra/selfheal) runs. Models and data of the frozen project are
# read only; every output, cache and merged checkpoint goes under $SH.
SH=${SH_ROOT:-/scratch/pabitra/selfheal}
AP=${AP_ROOT_RO:-/scratch/pabitra/actionproof}          # read only
PY=$AP/.venv/bin/python
cd "$SH/wt" || exit 1
case "$(pwd -P)" in "$AP"*) echo "refusing to run inside the frozen project"; exit 1;; esac
export PYTHONPATH="$SH/wt" HF_HOME=/scratch/pabitra/.cache/huggingface HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export TMPDIR=$SH/.cache/tmp VLLM_CACHE_ROOT=$SH/.cache/vllm TRITON_CACHE_DIR=$SH/.cache/triton
export FLASHINFER_WORKSPACE_BASE=$SH/.cache/flashinfer OUTLINES_CACHE_DIR=$SH/.cache/outlines
mkdir -p "$TMPDIR" "$VLLM_CACHE_ROOT" "$TRITON_CACHE_DIR" "$FLASHINFER_WORKSPACE_BASE" "$OUTLINES_CACHE_DIR" logs
unset CUDA_HOME LIBRARY_PATH LD_LIBRARY_PATH VLLM_ALLOW_RUNTIME_LORA_UPDATING
export VLLM_USE_FLASHINFER_SAMPLER=0 VLLM_ENABLE_V1_MULTIPROCESSING=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
_p=""; for _d in ${PATH//:/ }; do [ -d "$_d" ] && [ -x "$_d" ] && _p="${_p:+$_p:}$_d"; done; export PATH="$_p"; unset _p _d
$PY scripts/selfheal_run.py splits --check > logs/splits_check.log 2>&1 || { cat logs/splits_check.log; exit 1; }
LEDGER=$SH/results/ledger.json
EV=results/selfheal/eval
SEEDS="0 1 2"

inc_model() { echo "$AP/runs/p1_d_s$1/merged"; }           # incumbent D_s{k}
cand_dir() { echo "$SH/runs/sh_$1_c_s$2"; }                 # round candidate C_{round}_s{k}

ev() {  # ev <tag> <split> <samples> --model M [--adapter A]
  local tag=$1 split=$2 n=$3; shift 3
  [ -f $EV/${tag}__${split}.summary.json ] && { echo "skip $tag $split"; return 0; }
  case $split in sh_gate_*) ;; *) rm -f $EV/${tag}__${split}.jsonl;; esac
  $PY scripts/selfheal_eval.py --tag $tag --split $split --samples $n "$@" > logs/ev_${tag}_${split}.log 2>&1 \
    || { tail -30 logs/ev_${tag}_${split}.log; exit 1; }
  tail -1 logs/ev_${tag}_${split}.log
}

pool_eval() {  # 1. RUN: the current agent on the round's loop pools (round 1: the incumbent D)
  local r=$1
  for k in $SEEDS; do
    for sp in sh_pool_${r}_p1full sh_pool_${r}_p1rec sh_pool_${r}_txn; do
      if [ "$r" = r1 ]; then ev D_s$k $sp 4 --model $(inc_model $k)
      else ev C_r1_s$k $sp 4 --model $(inc_model $k) --adapter $(cand_dir r1 $k)/final; fi
    done
  done
}

correct_curate() {  # 2-4. DIAGNOSE, CORRECT (B4 teacher), CURATE, MIX
  local r=$1 tag=$2 O=$SH/results/$1
  mkdir -p $O
  local IN="$(ls $EV/${tag}_s?__sh_pool_${r}_*.jsonl)"
  $PY scripts/selfheal_run.py diagnose --out $O/diagnose --inputs $IN > logs/${r}_diag.log 2>&1 || exit 1
  $PY scripts/selfheal_run.py correct --teacher b4 --ledger $LEDGER --round $r --out $O/corrections.jsonl --inputs $IN \
    > logs/${r}_correct.log 2>&1 || { tail -30 logs/${r}_correct.log; exit 1; }
  $PY scripts/selfheal_run.py curate --ledger $LEDGER --round $r --out-dir $O/curated --corrections $O/corrections.jsonl \
    > logs/${r}_curate.log 2>&1 || { tail -30 logs/${r}_curate.log; exit 1; }
  $PY scripts/selfheal_run.py mix --rows $O/curated/sft_train.jsonl --replay $AP/data/rft/p1_dwarm_rows.jsonl --ratio 1.0 \
    --out $O/curated/sft_mix.jsonl || exit 1
  cat $O/corrections.stats.json $O/curated/stats.json | head -80
}

sft() {  # 5. TRAIN: SFT on corrections + replay, from the same-seed incumbent (P1 arm C settings)
  local r=$1 k=$2 base=$3 out=$(cand_dir $1 $2)
  [ -f $out/final/adapter_config.json ] || $PY -m actionproof.train.rft --data $SH/results/$r/curated/sft_mix.jsonl \
    --model $base --out $out --epochs 1 --lr 5e-5 --lora-r 16 --seed $k > logs/${r}_sft_s$k.log 2>&1 \
    || { tail -40 logs/${r}_sft_s$k.log; exit 1; }
  rm -rf $out/checkpoint-*/optimizer.pt
  $PY -m actionproof.train.merge $out/final $base > logs/${r}_merge_s$k.log 2>&1 || exit 1
}

gate_eval() {  # 6a. both arms on every gate split (1 greedy + 2 sampled episodes, as TXN1)
  local r=$1
  for k in $SEEDS; do
    for sp in sh_gate_p1full sh_gate_p1rec sh_gate_txn sh_gate_txn_ood sh_gate_p1wf4 sh_gate_p1combo; do
      ev D_s$k $sp 2 --model $(inc_model $k)
      ev C_${r}_s$k $sp 2 --model $(cand_dir $r $k)/merged
    done
  done
}

reference() {  # B4 on the gate splits (scripted, CPU), for the descriptive share of the D-B4 gap closed
  $PY scripts/selfheal_run.py reference --policy B4 --out-dir $EV \
    --splits sh_gate_p1full sh_gate_p1rec sh_gate_txn sh_gate_txn_ood sh_gate_p1wf4 sh_gate_p1combo || exit 1
}

gate() {  # 6b. promote/reject under the frozen criteria
  local r=$1 C="" I=""
  for k in $SEEDS; do C="$C $(ls $EV/C_${r}_s${k}__sh_gate_*.jsonl)"; I="$I $(ls $EV/D_s${k}__sh_gate_*.jsonl)"; done
  $PY scripts/selfheal_run.py gate --criteria configs/selfheal/gate_v1.json --ledger $LEDGER \
    --candidate $C --incumbent $I --reference $(ls $EV/B4__sh_gate_*.jsonl) --out $SH/results/$r/gate_decision.json || exit 1
}
