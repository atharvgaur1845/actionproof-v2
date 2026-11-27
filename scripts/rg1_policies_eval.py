"""RG1 scripted arms (PREREGISTRATION_RG1.md): the TXN baselines with and without the Recovery Gate, one episode per
task, on the canonical tool names (actionproof/txn/policies.py). CPU only.

    python scripts/rg1_policies_eval.py --split rg1_test
Writes results/rg1/<name>__<split>.jsonl and <name>_gate__<split>.jsonl (+ .summary.json) for B0, B1, B3, B4 and
the B4 ablations A1-A6 (no gate), and B0, B1, B3, B4 behind the full gate. Sealed splits run once, after the freeze
check.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
PLAIN = ("B0", "B1", "B2", "B3", "B4", "A1", "A2", "A3", "A4", "A5", "A6")
GATED = ("B0", "B1", "B3", "B4")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True, choices=("txn_dev", "txn_val", "txn_test", "rg1_test"))
    a = ap.parse_args()
    from actionproof.bench.generator_proc import DATA, load_tasks
    from actionproof.recovery_gate.env import run_policy_gated
    from actionproof.recovery_gate.gate import Rules
    from actionproof.txn.policies import run_policy
    from actionproof.txn.truth import breakdown, summarise
    if a.split in ("txn_test", "rg1_test"):
        if subprocess.run([sys.executable, str(ROOT / "scripts/freeze_rg1.py"), "--check"], cwd=ROOT).returncode:
            raise SystemExit("freeze check failed")
    tasks = load_tasks(DATA / f"{a.split}.jsonl")
    out_dir = ROOT / "results/rg1"
    out_dir.mkdir(parents=True, exist_ok=True)
    jobs = [(n, n, lambda t, n=n: run_policy(n, t)) for n in PLAIN]
    jobs += [(f"{n}_gate", n, lambda t, n=n: run_policy_gated(n, t, Rules())) for n in GATED]
    for tag, name, fn in jobs:
        out = out_dir / f"{tag}__{a.split}.jsonl"
        if out.exists():
            print("skip", tag)
            continue
        rows = []
        for t in tasks:
            r = fn(t)
            r["tag"] = tag
            rows.append(r)
        out.write_text("".join(json.dumps(r, default=str) + "\n" for r in rows))
        sc = [r["txn"] for r in rows]
        s = {"tag": tag, "policy": name, "gate": tag.endswith("_gate"), "split": a.split, "n_tasks": len(tasks),
             "txn": {"all": summarise(sc), "by_group": breakdown(sc, "group"), "by_regime": breakdown(sc, "regime")}}
        out.with_suffix(".summary.json").write_text(json.dumps(s, indent=2))
        al = s["txn"]["all"]
        print(tag.ljust(10), {k: f"{al[k]['num']}/{al[k]['den']}" for k in ("VTC", "STR", "FCR", "DMR_all", "CER")})


if __name__ == "__main__":
    main()
