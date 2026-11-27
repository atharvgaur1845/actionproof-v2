"""RG2 scripted arms on rg2_test (PREREGISTRATION_RG2.md; CPU): B4, B5 and always-escalate with no gate; B4 behind
gate v1 and v2; B5 behind gate v2. One episode per task. Writes results/rg2/<tag>__rg2_test.jsonl (+ .summary.json)."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
ARMS = (("B4", "B4", None), ("B5", "B5", None), ("ALWAYS", "ALWAYS", None), ("B4_v1", "B4", "v1"),
        ("B4_v2", "B4", "v2"), ("B5_v2", "B5", "v2"))


def main() -> None:
    from actionproof.bench.generator_proc import DATA, load_tasks
    from actionproof.recovery_gate.rg2_env import run_policy_rg2
    from actionproof.txn.truth import breakdown, summarise
    if subprocess.run([sys.executable, str(ROOT / "scripts/freeze_rg2.py"), "--check"], cwd=ROOT).returncode:
        raise SystemExit("freeze check failed")
    tasks = load_tasks(DATA / "rg2_test.jsonl")
    d = ROOT / "results/rg2"
    d.mkdir(parents=True, exist_ok=True)
    for tag, pol, gate in ARMS:
        out = d / f"{tag}__rg2_test.jsonl"
        if out.exists():
            print("skip", tag)
            continue
        rows = [run_policy_rg2(pol, t, gate) for t in tasks]
        out.write_text("".join(json.dumps(r, default=str) + "\n" for r in rows))
        sc = [r["txn"] for r in rows]
        out.with_suffix(".summary.json").write_text(json.dumps({"tag": tag, "policy": pol, "gate": gate,
                                                                "txn": {"all": summarise(sc),
                                                                        "by_group": breakdown(sc, "group")}}, indent=2))
        print(tag, "done")


if __name__ == "__main__":
    main()
