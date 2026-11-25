#!/usr/bin/env bash
# Verify the published numbers without a GPU. CPU only, from a public clone of the code repository.
#
#   bash scripts/verify_numbers.sh                 # download the traces, recompute, diff, check the claims ledger
#   SKIP_DOWNLOAD=1 bash scripts/verify_numbers.sh # episode files are already under results/ (e.g. copied from the cluster)
#
# What it does
#   1. downloads the episode files (results/{p1,p2,p3}/**/*.jsonl) of the dataset atharvgaur18/actionproof-traces into
#      results/. Only *.jsonl: the summaries stay the ones committed in git, so step 3 compares a recomputation against
#      the committed values. The frozen task splits (data/bench/proc/) are already in the clone.
#   2. reruns the registered analyses, in place:
#        scripts/p1_stats.py           -> results/p1/p1_summary.json     (P1: B, C, D versus A)
#        scripts/p2_stats.py           -> results/p2/p2_summary.json     (P2: coverage gate, then the contrasts)
#        scripts/p3_stats.py           -> results/p3/p3_summary.json     (P3: Dn versus D)
#        scripts/audit/dwarm_2x2.py    -> results/audit/dwarm_2x2.json   (exploratory W / A / Dw / D / Dn 2x2)
#      None of these refuses to overwrite its output, so they write over the committed file and git is the comparison.
#      Every bootstrap uses a fixed seed (20261124 or 20261125) and numpy's PCG64, so a rerun is deterministic.
#      p2_stats.py also writes results/p2/p2_coverage.json (untracked). It stops with exit code 2 if any required
#      episode file is missing or incomplete.
#   3. git diff --exit-code results/        (byte-identical to the committed results, else FAIL)
#   4. python scripts/s2_numcheck.py        (every number in the paper, write-up and README resolves through claims/claims.yaml)
#
# Needs: python >= 3.10 with numpy, pyyaml and huggingface_hub (pip install numpy pyyaml huggingface_hub), and git.
# Runtime: roughly 10 minutes, almost all of it the download (about 2 GB of episode files). The four analyses take a
# couple of minutes on one CPU core (they read the 2,250 episodes per model once each; p1_stats.py and p3_stats.py were
# timed at a few seconds with the files in the page cache). Memory: under 4 GB.
#
# Environment: PYTHON (default python), TRACES_REPO (default atharvgaur18/actionproof-traces), SKIP_DOWNLOAD=1,
# ALLOW_FLOAT_NOISE=1 (accept differences below 1e-9 relative, which a different numpy build could cause).
set -euo pipefail
cd "$(dirname "$0")/.."

PY="${PYTHON:-python}"
REPO="${TRACES_REPO:-atharvgaur18/actionproof-traces}"
step() { printf '\n== %s\n' "$*"; }

step "0. preconditions"
git rev-parse --is-inside-work-tree >/dev/null || { echo "run this from a git clone of the code repository"; exit 1; }
if ! git diff --quiet -- results/; then
  echo "results/ has uncommitted changes; the comparison would be meaningless. Stash them or use a fresh clone."
  exit 1
fi
"$PY" - <<'EOF'
import importlib.util, sys
missing = [m for m in ("numpy", "yaml") if importlib.util.find_spec(m) is None]
if missing:
    sys.exit(f"missing python packages: {missing} (pip install numpy pyyaml huggingface_hub)")
EOF

step "1. download episode files from $REPO"
if [ "${SKIP_DOWNLOAD:-0}" = "1" ]; then
  echo "SKIP_DOWNLOAD=1: using the episode files already under results/"
else
  REPO="$REPO" "$PY" - <<'EOF'
import os
from huggingface_hub import snapshot_download
snapshot_download(repo_id=os.environ["REPO"], repo_type="dataset", local_dir=".",
                  allow_patterns=["results/p1/**/*.jsonl", "results/p2/**/*.jsonl", "results/p3/**/*.jsonl"],
                  ignore_patterns=["**/smoke/**"])
EOF
fi
n=$(find results/p1/test -name '*.jsonl' | wc -l)
echo "results/p1/test: $n episode files (expected 80: A, B, C, D, Dn x 3 seeds + Dw, 5 splits each)"
[ "$n" -ge 80 ] || { echo "FAIL: episode files missing"; exit 1; }

step "2. rerun the analyses"
analyse() {  # run one analysis; on failure show its output and stop (a failure inside "&&" would not trip set -e)
  local out
  if ! out=$("$PY" "$1" 2>&1); then
    echo "FAIL: $1"; echo "$out" | tail -15; exit 1
  fi
  echo "$1 ok"
}
analyse scripts/p1_stats.py
analyse scripts/p2_stats.py
analyse scripts/p3_stats.py
analyse scripts/audit/dwarm_2x2.py

step "3. recomputed summaries versus the committed ones (git diff --exit-code results/)"
if git diff --exit-code --stat -- results/; then
  echo "identical to the committed results"
else
  echo "DIFFERENT from the committed results (files above)."
  if [ "${ALLOW_FLOAT_NOISE:-0}" = "1" ]; then
    "$PY" - <<'EOF' || { echo "FAIL: differences larger than float noise"; exit 1; }
import json, math, subprocess, sys
def close(a, b):
    if isinstance(a, float) or isinstance(b, float):
        return math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-12)
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(close(a[k], b[k]) for k in a)
    if isinstance(a, list):
        return len(a) == len(b) and all(close(x, y) for x, y in zip(a, b))
    return a == b
files = subprocess.check_output(["git", "diff", "--name-only", "--", "results/"], text=True).split()
bad = []
for f in files:
    old = json.loads(subprocess.check_output(["git", "show", f"HEAD:{f}"], text=True))
    if not close(old, json.load(open(f))):
        bad.append(f)
print("float-noise only" if not bad else f"real differences in {bad}")
sys.exit(1 if bad else 0)
EOF
    echo "ALLOW_FLOAT_NOISE=1: continuing"
  else
    echo "FAIL. Inspect with: git diff results/   (restore with: git checkout -- results/)"
    exit 1
  fi
fi

step "4. claims ledger (scripts/s2_numcheck.py)"
"$PY" scripts/s2_numcheck.py

printf '\nPASS: the recomputed analyses match the committed results and every claim resolves.\n'
