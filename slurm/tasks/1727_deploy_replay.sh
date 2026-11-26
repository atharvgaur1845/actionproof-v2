# minutes: 30
# Redeploy the static replay Space with the committed demo/static/index.html. Network only.
set -uo pipefail
source slurm/claim.sh || exit 0
source slurm/env.sh
unset HF_HUB_OFFLINE
.venv/bin/python scripts/deploy_replay_space.py || exit 1
echo DEPLOY_REPLAY_OK
