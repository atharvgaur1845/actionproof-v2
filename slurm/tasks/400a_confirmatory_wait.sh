# minutes: 480
# Confirmatory H1 (task 400). Waits (≤ 6 h) for G3 instead of failing, so an early claim does not trip the worker's
# two-failure circuit breaker. No dependency on A5 reruns (held, Amendment 6 §6).
set -euo pipefail
for i in $(seq 36); do [ "$(date +%Y%m%d)" -ge 20261111 ] && break; sleep 600; done
bash slurm/tasks/400_confirmatory_eval.sh
