#!/bin/bash
# Poll the cluster until at least N files match a pattern under results/ad/dev (or a failed AD task appears).
#   bash scripts/ad/wait_remote.sh <grep pattern> <N> <max minutes>
PAT=$1; N=$2; MAXM=${3:-10}
for ((i = 0; i < MAXM; i++)); do
  out=$(timeout 60 ssh -o LogLevel=ERROR sharanga "ls /scratch/pabitra/actionproof/results/ad/dev/ | grep -E '$PAT' | wc -l; ls /scratch/pabitra/actionproof/queue/*/failed/ | grep '^12' | wc -l" 2>/dev/null | grep -E '^[0-9]+$' | tr '\n' ' ')
  set -- $out
  if [ "${1:-0}" -ge "$N" ] || [ "${2:-0}" -ge 1 ]; then echo "$(date +%H:%M) matched=$1 failed=$2"; exit 0; fi
  sleep 60
done
echo "$(date +%H:%M) not yet (matched=${1:-?})"
