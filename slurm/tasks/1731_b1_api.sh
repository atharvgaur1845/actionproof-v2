# minutes: 240
# B1 addendum 1: Qwen3-235B and gpt-oss-120b via the HF router, greedy, P1 test; hard stop at $7 (scripts/b1_api_eval.py). Network only.
set -uo pipefail
source slurm/claim.sh || exit 0
source slurm/env.sh
for m in qwen gptoss; do for sp in p1_test_full p1_test_pp p1_test_rec p1_test_wf4 p1_test_combo; do
  [ -f results/b1/${m}__${sp}.summary.json ] && continue
  .venv/bin/python scripts/b1_api_eval.py --model $m --split $sp --cap 7 || exit 1
done; done
echo B1_API_OK
