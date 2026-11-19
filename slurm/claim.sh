# Cross-pool claim (Stage 2). Copies of one task sit in several pools (h200, a100, h100) and may also be launched
# on HF Jobs; the first executor to create queue/claims/<task> runs it, every other copy deletes itself and exits.
# A failed run releases its claim so another executor can retry. On HF Jobs (AP_HF=1) the launcher claimed already.
#   usage, first line of a task body:   source slurm/claim.sh || exit 0
AP_TASK=$(basename "$0" .sh)
if [ -z "${AP_HF:-}" ]; then
  mkdir -p queue/claims
  if ! mkdir "queue/claims/$AP_TASK" 2>/dev/null; then
    echo "claimed elsewhere: $(cat "queue/claims/$AP_TASK/owner" 2>/dev/null)"
    rm -f "$0"
    return 1
  fi
  echo "slurm:${SLURM_JOB_ID:-?}:$(hostname):$(date -Is)" > "queue/claims/$AP_TASK/owner"
  trap 'rc=$?; if [ $rc -ne 0 ]; then rm -rf "queue/claims/$AP_TASK"; else date -Is > "queue/claims/$AP_TASK/done"; fi' EXIT
fi
return 0
