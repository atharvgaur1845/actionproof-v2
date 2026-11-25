# minutes: 300
# P3 (PREREGISTRATION_P3.md): arm Dn seed 1: P1 arm D's command with scale_rewards="none"; then dev and frozen-test evaluation.
set -uo pipefail
source slurm/claim.sh || exit 0
source slurm/env.sh
.venv/bin/python scripts/freeze_p3.py --check || exit 1
export MASTER_PORT=29961 AP_ROLLOUT_LOG=results/p3/rollouts/Dn_s1.jsonl
mkdir -p results/p3/rollouts
[ -f runs/p3_dn_s1/final/adapter_config.json ] || .venv/bin/python -m actionproof.p3.train_nostd --reward rv \
  --pools data/bench/proc/s3_trainb_half.jsonl,data/bench/proc/s3_rec_train_half.jsonl --seed 1 --steps 200 \
  --init-adapter runs/p1_dwarm/final --out runs/p3_dn_s1 --use-vllm > logs/p3_dn_s1.full.log 2>&1 || { tail -40 logs/p3_dn_s1.full.log; exit 1; }
grep -q '"scale_rewards": "none"' runs/p3_dn_s1/run_config.json 2>/dev/null || grep -q "scale_rewards='none'\|scale_rewards=none" logs/p3_dn_s1.full.log || echo "note: scale_rewards not echoed in run_config (set by the wrapper)"
gzip -f $AP_ROLLOUT_LOG; rm -rf runs/p3_dn_s1/checkpoint-*/optimizer.pt
source slurm/p1_evalset.sh dev "Dn_s1|runs/p1_dwarm/merged|runs/p3_dn_s1/final|0" "p1_dev_full p1_dev_rec"
source slurm/p1_evalset.sh test "Dn_s1|runs/p1_dwarm/merged|runs/p3_dn_s1/final|0" "p1_test_full p1_test_pp p1_test_rec p1_test_wf4 p1_test_combo"
echo P3_DN_S1_OK
