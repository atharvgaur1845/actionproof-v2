#!/usr/bin/env bash
# One-time environment setup on sharanga's login node (no GPU needed to install).
# Pins the three Lexsi toolkits to the SHAs recorded in the spec so results are reproducible.
#   AgentTune  d4db4c12d523  -> torch 2.11.0, trl[vllm] 1.7.1 (vllm 0.22.1), transformers>=5.15,<6
#   AuditKIT   71316da34578  -> installed WITHOUT its [vllm] extra (wants vllm>=0.30, conflicts with AgentTune)
#   CuratorKIT dd9e835301fc  -> core + [hf]
# SAE weights are loaded with our own JumpReLU encoder (actionproof/interp/sae_features.py),
# not sae-lens, to keep torch pinned at AgentTune's version.
set -euo pipefail
ROOT=${ROOT:-$HOME/actionproof}
cd "$ROOT"

declare -A SHAS=( [AgentTune]=d4db4c12d523 [AuditKIT]=71316da34578 [CuratorKIT]=dd9e835301fc )
mkdir -p vendor
for repo in "${!SHAS[@]}"; do
  [ -d "vendor/$repo" ] || git clone -q "https://github.com/Lexsi-Labs/$repo.git" "vendor/$repo"
  git -C "vendor/$repo" fetch -q origin
  git -C "vendor/$repo" checkout -q "${SHAS[$repo]}"
  echo "$repo @ $(git -C vendor/$repo rev-parse --short=12 HEAD)"
done

uv venv --python 3.12 .venv
# shellcheck disable=SC1091
source .venv/bin/activate
uv pip install -e "vendor/AgentTune[vllm,dev]"
uv pip install -e "vendor/AuditKIT[transformers,interop]"
uv pip install -e "vendor/CuratorKIT[hf]"
uv pip install -e ".[dev]"

python - <<'EOF'
import importlib, torch
print("torch", torch.__version__, "cuda", torch.version.cuda)
for mod in ["transformers", "trl", "vllm", "peft", "agenttune", "auditkit", "curatorkit"]:
    try:
        m = importlib.import_module(mod)
        print(f"{mod:12s}", getattr(m, "__version__", "?"))
    except Exception as e:  # report, don't abort: the log is the artifact
        print(f"{mod:12s} IMPORT FAILED: {type(e).__name__}: {e}")
EOF
uv pip freeze > requirements.lock
echo "SETUP_DONE"
