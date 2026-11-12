"""Toolkit chain on the final files (Phase 3): run `actionproof.eval.auditkit_bridge` on every confirmatory
(`results/eval/conf/*_c.jsonl`) and headline development (`results/eval/r3`, 4 arms x 2 seeds x 2 splits) file and
check that AuditKIT's verified-success rate equals our VSC on each.

    python scripts/auditkit_bridge_final.py > results/auditkit/bridge_final.md
"""
from __future__ import annotations

import glob
import json
import re
from pathlib import Path

from actionproof.eval.auditkit_bridge import run

FILES = sorted(glob.glob("results/eval/conf/*_c.jsonl")) + sorted(
    p for p in glob.glob("results/eval/r3/*.jsonl")
    if re.search(r"grpo_tiny_v2_(pcstr_d|outcome|tool_trust|ack_outcome)_s[01]__neutral__test_(iid|ood_fault)\.jsonl$", p))

print("| file | n | our VSC (sum) | AuditKIT verified pass | match |\n|---|---|---|---|---|")
ok = 0
for p in FILES:
    recs = [json.loads(l) for l in open(p)]
    ours = sum(r["vsc"] for r in recs)
    run(p)
    res = json.load(open(f"results/auditkit/{Path(p).stem}.auditkit.json"))
    rows = res["rows"] if isinstance(res, dict) else res
    ak = sum(r["outcome"]["verdict"] == "success" for r in rows)
    ok += ak == ours
    print(f"| {Path(p).stem} | {len(recs)} | {ours:g} | {ak} | {'yes' if ak == ours else '**no**'} |")
print(f"\n{ok}/{len(FILES)} files: AuditKIT's verified outcome reproduces VSC exactly.")
