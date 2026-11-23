# minutes: 40
# Quality audit (docs/quality/EXPERIMENT_PIPELINE_AUDIT.md §D; descriptive, not a registered test): G's sampler/scorer gap on its hinted rollouts
set -uo pipefail
source slurm/claim.sh || exit 0
source slurm/env.sh
mkdir -p results/quality
.venv/bin/python scripts/audit_g_mismatch.py --log results/s5/rollouts/s5_guide_s0.jsonl.gz --model runs/rft_proc/merged \
  --adapter runs/s5_guide_s0/checkpoint-100 --out results/quality/g_mismatch_s0_ck100.json > logs/g_audit.log 2>&1 || { tail -30 logs/g_audit.log; exit 1; }
tail -15 logs/g_audit.log
