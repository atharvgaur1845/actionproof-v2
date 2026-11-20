# minutes: 60
# A16 exploratory CircuitKIT EAP-IG on R_V seed 0, 20 matched dev pairs (PREREGISTRATION_A15 §5; never used for a claim).
set -uo pipefail
source slurm/claim.sh || exit 0
source slurm/env.sh
mkdir -p logs results/a16/circuitkit
git clone --depth 1 https://github.com/Lexsi-Labs/CircuitKIT /tmp/ck > logs/ck_clone.log 2>&1 || exit 1
uv pip install -q --python .venv /tmp/ck > logs/ck_install.log 2>&1 || { tail -20 logs/ck_install.log; exit 1; }
[ -f runs/rft_proc/merged/config.json ] || .venv/bin/python -m actionproof.train.merge runs/rft_proc/final \
  CohereLabs/tiny-aya-global > logs/ck_merge.log 2>&1 || exit 1
[ -f runs/a12_proc_rv_s0/merged/config.json ] || .venv/bin/python -c "from actionproof.train.merge import merge; merge('runs/a12_proc_rv_s0/final', 'runs/rft_proc/merged')" > logs/ck_merge2.log 2>&1 || { tail -20 logs/ck_merge2.log; exit 1; }
timeout 50m .venv/bin/python scripts/a16_circuitkit.py --weights runs/a12_proc_rv_s0/merged > logs/a16_ck.log 2>&1
tail -40 logs/a16_ck.log
rm -rf runs/a12_proc_rv_s0/merged
