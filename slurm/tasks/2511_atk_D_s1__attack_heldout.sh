# minutes: 150
# DRAFT, NOT QUEUED. Attack suite ATK1 (docs/attacks/SPEC.md, PREREGISTRATION_ATTACKS_DRAFT.md): D_s1 on attack_heldout,
# without the Recovery Gate (rules none) and with it (rules full); greedy + 2 samples per task. GPU (vLLM).
# Runs in /scratch/pabitra/attacks (a copy of branch research/attack-suite) with caches there; reads the merged model
# from the frozen tree and writes nothing under /scratch/pabitra/actionproof.
# Needs a frozen PREREGISTRATION_ATTACKS.md first (scripts/attacks_eval.py refuses otherwise).
set -uo pipefail
cd /scratch/pabitra/attacks || exit 1
export AP_ROOT=/scratch/pabitra/attacks
source slurm/env.sh
bash slurm/attacks/atk_run.sh NG_D_s1 attack_heldout none /scratch/pabitra/actionproof/runs/p1_d_s1/merged || exit 1
bash slurm/attacks/atk_run.sh RG_D_s1 attack_heldout full /scratch/pabitra/actionproof/runs/p1_d_s1/merged || exit 1
echo ATK_D_s1__attack_heldout_OK
