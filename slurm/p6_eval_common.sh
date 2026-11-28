# Shared P6 evaluation body: one arm and one seed on all seven registered splits.
# Source from a task wrapper after slurm/claim.sh and slurm/env.sh.
run_p6_eval() {
  local arm="$1" seed="$2" model tag split
  .venv/bin/python scripts/freeze_p6.py --check || return 1
  if [ "$arm" = "canonical" ]; then
    model="runs/p6_canonical_s${seed}/merged"
    tag="P6C_s${seed}"
  elif [ "$arm" = "surface" ]; then
    model="runs/p6_surface_s${seed}/merged"
    tag="P6S_s${seed}"
  else
    echo "unknown P6 arm: $arm" >&2; return 2
  fi
  [ -f "$model/config.json" ] || { echo "missing merged model: $model" >&2; return 1; }
  mkdir -p logs
  for split in s4_api s4_wf4 p1_test_full p1_test_pp p1_test_rec p1_test_wf4 p1_test_combo; do
    if [ -f "results/p6/${tag}__${split}.jsonl" ]; then
      echo "already evaluated: ${tag}__${split}"
      continue
    fi
    .venv/bin/python scripts/p6_eval.py --model "$model" --tag "$tag" --split "$split" --samples 4 \
      > "logs/${tag}__${split}.log" 2>&1 || { tail -60 "logs/${tag}__${split}.log"; return 1; }
  done
}
