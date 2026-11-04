# minutes: 30
# Upgrade 4: naturalistic F8 acknowledgements (Aya 32B + CuratorKIT gates), frozen into the repo afterwards.
set -euo pipefail
source slurm/env.sh
.venv/bin/python -m actionproof.bench.naturalistic_acks --out actionproof/bench/data/f8_natural.json > logs/172_f8.full.log 2>&1
tail -1 logs/172_f8.full.log
python3 -c "import json;d=json.load(open('actionproof/bench/data/f8_natural.json'));[print(k, v[:3]) for k,v in d['texts'].items()]"
