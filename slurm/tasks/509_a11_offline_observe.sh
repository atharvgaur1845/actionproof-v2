# minutes: 40
# Amendment 11 (PREREGISTRATION_A11, H1): Observe mode computed offline (CPU) from every passthrough episode file of
# tasks 500 to 506. Each recorded call sequence is replayed in a fresh environment through an observe gateway; the
# run aborts on any replayed observation that differs from the recorded one. Then the scripted-agent reference runs.
# Not executed when written (2026-11-20); runs after 500 to 506.
set -uo pipefail
source slurm/env.sh
.venv/bin/python scripts/freeze_a11.py --check || exit 1
mkdir -p results/a11/eval results/a11/scripted logs
for p in results/a11/eval/*__passthrough__*.jsonl; do
  o=${p/__passthrough__/__observe__}; sp=${p##*__}; sp=${sp%.jsonl}
  [ -f ${o%.jsonl}.summary.json ] || .venv/bin/python scripts/eval_gateway.py --offline-observe $p --split $sp --out $o \
    > logs/a11_observe_$(basename ${o%.jsonl}).full.log 2>&1 || echo "FAILED $o"
done
for ag in task_oracle lean_expert lazy_liar blind_retrier over_escalator no_op; do
  for mode in passthrough observe enforce enforce_inform; do
    for sp in test_iid_c2 v2_stress_c2; do
      o=results/a11/scripted/${ag}__${mode}__${sp}.jsonl
      [ -f ${o%.jsonl}.summary.json ] || .venv/bin/python scripts/eval_gateway.py --agent $ag --mode $mode --split $sp \
        --out $o > /dev/null || echo "FAILED $o"
    done
  done
done
ls results/a11/eval/*__observe__*.summary.json | wc -l
