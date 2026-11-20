# minutes: 45
# CircuitKIT smoke test (PREREGISTRATION_A15 §5, exploratory; user-approved install inside the job): install from
# GitHub, report the package layout, README and any EAP-IG / attribution entry points, and check it loads the merged
# warm start. Never used for a claim.
set -uo pipefail
source slurm/claim.sh || exit 0
source slurm/env.sh
mkdir -p logs results/a16/circuitkit
O=results/a16/circuitkit
git clone --depth 1 https://github.com/Lexsi-Labs/CircuitKIT /tmp/ck > $O/clone.log 2>&1 || { cat $O/clone.log; exit 1; }
( cd /tmp/ck && git log -1 --format='%H %cd' ) > $O/commit.txt
find /tmp/ck -maxdepth 3 -name "*.py" | sed 's#/tmp/ck/##' | sort > $O/files.txt
head -200 /tmp/ck/README.md > $O/README_head.md 2>/dev/null
grep -rniE "eap|edge.attribution|integrated.gradient|tiny.aya|cohere" /tmp/ck --include=*.py --include=*.md | head -80 > $O/grep.txt
uv pip install -q --python .venv /tmp/ck > $O/install.log 2>&1; echo "install rc=$?" >> $O/install.log
[ -f runs/rft_proc/merged/config.json ] || .venv/bin/python -m actionproof.train.merge runs/rft_proc/final \
  CohereLabs/tiny-aya-global > logs/ck_merge.log 2>&1
.venv/bin/python - > $O/import.txt 2>&1 <<'PY'
import importlib, pkgutil
for name in ("circuitkit", "CircuitKIT", "circuit_kit"):
    try:
        m = importlib.import_module(name)
        print("imported", name, getattr(m, "__version__", ""))
        print([x.name for x in pkgutil.walk_packages(m.__path__, name + ".")][:80])
        print(sorted(a for a in dir(m) if not a.startswith("_")))
        break
    except Exception as e:
        print("no", name, type(e).__name__, e)
PY
cat $O/commit.txt $O/install.log | tail -5; head -20 $O/import.txt
