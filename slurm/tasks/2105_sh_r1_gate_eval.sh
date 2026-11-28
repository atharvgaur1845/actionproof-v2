# minutes: 600
# Self-heal round 1, PREPARED, NOT SUBMITTED: incumbent and candidate on the fresh gate splits (each run once).
# after: 2102_sh_r1_sft_s0 2103_sh_r1_sft_s1 2104_sh_r1_sft_s2
set -uo pipefail
source wt/slurm/claim.sh || exit 0
source wt/slurm/selfheal/round.sh
gate_eval r1
echo SH_R1_GATE_EVAL_OK
