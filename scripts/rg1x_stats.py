"""RG1 addendum 1, descriptive table: per sel1_test group, the frozen gate's arms (results/rg1/X1_*__sel1_test.jsonl)
next to the same models without the gate (results/sel1/D_s0, SEL_s0) and the scripted references (B4, B5, ALWAYS).
Rates are over episodes (3 per task for models, 1 for scripts), with numerators and denominators.

    python scripts/rg1x_stats.py      # writes results/rg1/rg1x_summary.json and .md
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GROUPS = ("u_in", "u_receive", "u_ood_wf", "t2", "audit_down", "std")
ARMS = [("D seed 0, no gate", "results/sel1/D_s0__sel1_test.jsonl"),
        ("D seed 0, gate", "results/rg1/X1_RG_D_s0__sel1_test.jsonl"),
        ("SEL seed 0, no gate", "results/sel1/SEL_s0__sel1_test.jsonl"),
        ("SEL seed 0, gate", "results/rg1/X1_RG_SEL_s0__sel1_test.jsonl"),
        ("B4 (script)", "results/sel1/policy_B4__sel1_test.jsonl"),
        ("B5 (selective script)", "results/sel1/policy_B5__sel1_test.jsonl"),
        ("always escalate", "results/sel1/policy_ALWAYS__sel1_test.jsonl")]
FLAGS = ("escalated", "str", "vtc", "unsafe", "fcr")


def main() -> None:
    out: dict = {}
    for name, path in ARMS:
        f = ROOT / path
        if not f.exists():
            out[name] = None
            continue
        by: dict[str, list] = defaultdict(list)
        for line in f.read_text().splitlines():
            if line.strip():
                t = json.loads(line)["txn"]
                by[t["group"]].append(t)
        out[name] = {g: {fl: [sum(bool(e[fl]) for e in by[g]), len(by[g])] for fl in FLAGS} for g in GROUPS}
    (ROOT / "results/rg1/rg1x_summary.json").write_text(json.dumps(out, indent=1) + "\n")
    L = ["# RG1 addendum 1: the frozen gate on sel1_test (descriptive)", "",
         "Cells: escalated / STR / unsafe, as counts over episodes. Groups u_*: unresolvable (hand-off is correct); "
         "t2: the record returns after two reads; audit_down: only the audit log is down; std: standard.", "",
         "| Arm | " + " | ".join(GROUPS) + " |", "|---|" + "---|" * len(GROUPS)]
    for name, d in out.items():
        if d is None:
            L.append(f"| {name} | " + " | ".join("missing" for _ in GROUPS) + " |")
            continue
        cells = []
        for g in GROUPS:
            c = d[g]
            cells.append(f"esc {c['escalated'][0]}/{c['escalated'][1]}, STR {c['str'][0]}, unsafe {c['unsafe'][0]}")
        L.append(f"| {name} | " + " | ".join(cells) + " |")
    (ROOT / "results/rg1/rg1x_summary.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
