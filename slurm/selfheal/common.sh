# Sourced by the self-heal CPU jobs. Code runs from the self-heal copy; results of the frozen project are read only.
SH=${SH_ROOT:-/scratch/pabitra/selfheal}
R=${AP_RESULTS:-/scratch/pabitra/actionproof/results}   # read only
PY=${AP_PY:-/scratch/pabitra/actionproof/.venv/bin/python}  # read only
cd "$SH/wt"
export PYTHONPATH="$SH/wt" HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
mkdir -p "$SH/results" "$SH/logs"
case "$(pwd -P)" in /scratch/pabitra/actionproof*) echo "refusing to run inside the frozen project"; exit 1;; esac
