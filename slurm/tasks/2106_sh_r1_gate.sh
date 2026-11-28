# minutes: 30
# Self-heal round 1, PREPARED, NOT SUBMITTED: promote/reject decision under configs/selfheal/gate_v1.json (CPU).
# after: 2105_sh_r1_gate_eval
set -uo pipefail
source wt/slurm/claim.sh || exit 0
source wt/slurm/selfheal/round.sh
gate r1
cat $SH/results/r1/gate_decision.json | head -60
echo SH_R1_GATE_OK
