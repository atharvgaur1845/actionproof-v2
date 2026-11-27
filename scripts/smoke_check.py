"""Smoke check: about 15-30 s on one CPU core (27 s in both clean rooms), no download, no GPU, no token. A SMOKE TEST, not result reproduction:
it re-derives the benchmark and the scripted baselines from code and checks every published number's ledger, but it
does not recompute any model result (that is `bash scripts/verify_numbers.sh`, which downloads the episode files).

    pip install -e ".[dev]"        # numpy, scipy, pydantic, pytest, hypothesis, pyyaml
    python scripts/smoke_check.py

Checks:
1. the TXN and SEL1 test suites (`tests/txn`, `tests/sel1`): env semantics, goldens, the 25 scoring mutants,
   scripted references, the registered SEL1 analysis on scripted stand-ins;
2. the locked TXN splits and the SEL1 splits regenerate byte-identically from their seeds (sha256 = the manifests);
3. every scripted TXN baseline (B0-B4, A1-A6) rerun on the locked txn_test reproduces its committed summary
   (VTC, STR, FCR, DMR, URR, CER, with numerators and denominators);
4. `scripts/s2_numcheck.py`: every number in the scanned documents resolves through `claims/claims.yaml`.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
FAILS: list[str] = []


def step(name: str, ok: bool, detail: str = "") -> None:
    print(f"[{'PASS' if ok else 'FAIL'}] {name}{(': ' + detail) if detail else ''}", flush=True)
    if not ok:
        FAILS.append(name)


def main() -> None:
    t0 = time.time()
    r = subprocess.run([sys.executable, "-m", "pytest", "-q", "tests/txn", "tests/sel1"], cwd=ROOT,
                       capture_output=True, text=True)
    step("TXN + SEL1 test suites", r.returncode == 0, (r.stdout.strip().splitlines() or ["no output"])[-1])

    from actionproof.bench.generator_proc import DATA, load_tasks
    from actionproof.sel1 import tasks as SELT
    from actionproof.txn import tasks as T
    man = json.loads((DATA / "txn_manifest.json").read_text())
    for s in ("txn_dev", "txn_val", "txn_test"):
        h = hashlib.sha256("".join(T.task_line(t) for t in T.make_split(s)).encode()).hexdigest()
        step(f"{s} regenerates from seed {T.SEEDS[s]}", h == man["files"][f"{s}.jsonl"]["sha256"], h[:16])
    man = json.loads((DATA / "sel1_manifest.json").read_text())
    for s in SELT.SEEDS:
        h = hashlib.sha256("".join(T.task_line(t) for t in SELT.make_split(s)).encode()).hexdigest()
        step(f"{s} regenerates from seed {SELT.SEEDS[s]}", h == man["files"][f"{s}.jsonl"]["sha256"], h[:16])

    from actionproof.txn.policies import POLICIES, run_policy
    from actionproof.txn.truth import summarise
    tasks = load_tasks(DATA / "txn_test.jsonl")
    for name in POLICIES:
        got = summarise([run_policy(name, t)["txn"] for t in tasks])
        want = json.loads((ROOT / f"results/txn/{name}__txn_test.summary.json").read_text())["txn"]["all"]
        keys = ("VTC", "STR", "FCR", "DMR_all", "URR_all", "CER")
        same = all(got[k]["num"] == want[k]["num"] and got[k]["den"] == want[k]["den"] for k in keys)
        step(f"scripted {name} on txn_test", same,
             f"VTC {got['VTC']['num']}/{got['VTC']['den']}, FCR {got['FCR']['num']}/{got['FCR']['den']}")

    r = subprocess.run([sys.executable, "scripts/s2_numcheck.py"], cwd=ROOT, capture_output=True, text=True)
    step("claims ledger (s2_numcheck)", r.returncode == 0, (r.stdout.strip().splitlines() or ["no output"])[-1])
    print(f"\n{'ALL PASSED' if not FAILS else 'FAILED: ' + ', '.join(FAILS)} in {time.time() - t0:.0f} s "
          "(smoke test only; model results: bash scripts/verify_numbers.sh)")
    sys.exit(1 if FAILS else 0)


if __name__ == "__main__":
    main()
