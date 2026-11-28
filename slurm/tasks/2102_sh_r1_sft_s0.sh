# minutes: 150
# Self-heal round 1, PREPARED, NOT SUBMITTED: SFT candidate C_r1_s0 from incumbent D_s0 (1 epoch, lr 5e-5, LoRA r16).
# after: 2101_sh_r1_correct
set -uo pipefail
source wt/slurm/claim.sh || exit 0
source wt/slurm/selfheal/round.sh
sft r1 0 $(inc_model 0)
echo SH_R1_SFT_S0_OK
