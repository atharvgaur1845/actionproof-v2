# minutes: 150
# DRAFT, NOT QUEUED. Attack suite ATK1 (docs/attacks/SPEC.md, PREREGISTRATION_ATTACKS_DRAFT.md): base on attack_heldout,
# without the Recovery Gate (rules none) and with it (rules full); greedy + 2 samples per task. GPU (vLLM).
# Runs in /scratch/pabitra/attacks (a copy of branch research/attack-suite) with caches there; reads the merged model
# from the frozen tree and writes nothing under /scratch/pabitra/actionproof.
# Needs a frozen PREREGISTRATION_ATTACKS.md first (scripts/attacks_eval.py refuses otherwise).
set -uo pipefail
cd /scratch/pabitra/attacks || exit 1
export AP_ROOT=/scratch/pabitra/attacks
source slurm/env.sh
bash slurm/attacks/atk_run.sh NG_base attack_heldout none CohereLabs/tiny-aya-global || exit 1
bash slurm/attacks/atk_run.sh RG_base attack_heldout full CohereLabs/tiny-aya-global || exit 1
echo ATK_base__attack_heldout_OK
