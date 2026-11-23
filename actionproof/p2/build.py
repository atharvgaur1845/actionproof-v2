"""Arm E's warm-start data (PREREGISTRATION_P2.md §3): arm D's exact rows (`data/rft/p1_dwarm_rows.jsonl`
= `proc_demo_rows.jsonl` + `p1_recovery_rows.jsonl`), with the observable ledger added to every tool-result message
in each prompt, as the policy will see it in RL and evaluation. Targets (completions) are byte-identical, so E and D
differ only in the ledger text.

    python -m actionproof.p2.build
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from actionproof.p2.ledger import with_ledger

ROOT = Path(__file__).resolve().parents[2]
SRC = [ROOT / "data/rft/p1_dwarm_rows.jsonl"]  # = proc_demo_rows + p1_recovery_rows, built by task 1153
OUT = ROOT / "data/rft/p2_ewarm_rows.jsonl"


def ledgerise(prompt: list[dict]) -> list[dict]:
    out = []
    for i, m in enumerate(prompt):
        if i >= 2 and m["role"] == "user" and m["content"].startswith("Tool result for "):
            m = {**m, "content": with_ledger(m["content"], out)}
        out.append(m)
    return out


def main() -> None:
    n, h_in, h_out = 0, hashlib.sha256(), hashlib.sha256()
    with OUT.open("w") as f:
        for src in SRC:
            for line in src.read_text().splitlines():
                r = json.loads(line)
                h_in.update(json.dumps(r["completion"], sort_keys=True).encode())
                r2 = {**r, "prompt": ledgerise(r["prompt"])}
                h_out.update(json.dumps(r2["completion"], sort_keys=True).encode())
                f.write(json.dumps(r2) + "\n")
                n += 1
    assert h_in.hexdigest() == h_out.hexdigest()
    stats = {"rows": n, "sources": [str(s.relative_to(ROOT)) for s in SRC], "completions_sha256": h_out.hexdigest()}
    OUT.with_suffix(".stats.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats))


if __name__ == "__main__":
    main()
