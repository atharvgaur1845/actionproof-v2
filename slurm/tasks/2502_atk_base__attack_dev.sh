# minutes: 40
# DRAFT, NOT QUEUED. Attack suite ATK1 (docs/attacks/SPEC.md, PREREGISTRATION_ATTACKS_DRAFT.md): base on attack_dev,
# without the Recovery Gate (rules none) and with it (rules full); greedy + 2 samples per task. GPU (vLLM).
# Runs in /scratch/pabitra/attacks (a copy of branch research/attack-suite) with caches there; reads the merged model
# from the frozen tree and writes nothing under /scratch/pabitra/actionproof.
# Development pilot: no held-out data.
set -uo pipefail
cd /scratch/pabitra/attacks || exit 1
export AP_ROOT=/scratch/pabitra/attacks
source slurm/env.sh
bash slurm/attacks/atk_run.sh NG_base attack_dev none CohereLabs/tiny-aya-global || exit 1
bash slurm/attacks/atk_run.sh RG_base attack_dev full CohereLabs/tiny-aya-global || exit 1
echo ATK_base__attack_dev_OK
