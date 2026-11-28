# Shared P6 train body. Source from a task wrapper after slurm/claim.sh and slurm/env.sh.
run_p6_train() {
  local arm="$1" seed="$2" init="runs/p4_dwaug_s${2}/final"
  local out log mode
  if [ "$arm" = "canonical" ]; then
    out="runs/p6_canonical_s${seed}"
    mode="canonical"
  elif [ "$arm" = "surface" ]; then
    out="runs/p6_surface_s${seed}"
    mode="surface-random"
  else
    echo "unknown P6 arm: $arm" >&2; return 2
  fi
  log="logs/p6_${arm}_s${seed}.log"
  mkdir -p logs
  .venv/bin/python scripts/freeze_p6.py --check || return 1
  [ -f "$init/adapter_config.json" ] || { echo "missing P4 initial adapter: $init" >&2; return 1; }
  if [ -f "$out/merged/config.json" ]; then
    echo "already complete: $out/merged"
    return 0
  fi
  export AP_ROLLOUT_LOG="logs/p6_${arm}_s${seed}_rollouts.jsonl"
  if [ "$mode" = "canonical" ]; then
    .venv/bin/python -m actionproof.stage3.train \
      --reward rv --pools data/bench/proc/s3_trainb_half.jsonl,data/bench/proc/s3_rec_train_half.jsonl \
      --seed "$seed" --steps 200 --init-adapter "$init" --out "$out" --use-vllm --vllm-mem 0.30 \
      > "$log" 2>&1 || { tail -60 "$log"; return 1; }
  else
    .venv/bin/python -m actionproof.stage4.train_variants --surface-random \
      --reward rv --pools data/bench/proc/s3_trainb_half.jsonl,data/bench/proc/s3_rec_train_half.jsonl \
      --seed "$seed" --steps 200 --init-adapter "$init" --out "$out" --use-vllm --vllm-mem 0.30 \
      > "$log" 2>&1 || { tail -60 "$log"; return 1; }
  fi
  rm -rf "$out"/checkpoint-*/optimizer.pt
  .venv/bin/python -m actionproof.train.merge "$out/final" CohereLabs/tiny-aya-global \
    > "logs/p6_${arm}_s${seed}_merge.log" 2>&1 || { tail -60 "logs/p6_${arm}_s${seed}_merge.log"; return 1; }
  [ -f "$out/merged/config.json" ] || { echo "merge did not create $out/merged/config.json" >&2; return 1; }
}
