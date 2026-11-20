"""Exploratory: does GRPO change the greedy behaviour? Per confirmatory split and seed, the number of tasks whose
tool-call sequence (tool names and arguments, ignoring the free-text report summary) is identical to the warm start's
(results/eval/a12/conf_reference, evaluated after all models were frozen).
    python scripts/a12_identity.py --out results/s2/a12_identity.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def seqs(path: Path) -> dict[str, list]:
    out = {}
    for r in map(json.loads, path.open()):
        out[r["task_id"]] = [(s["tool"], json.dumps({k: v for k, v in (s["args"] or {}).items() if k != "summary"},
                                                   sort_keys=True)) for s in r["trace"]]
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/s2/a12_identity.json")
    a = ap.parse_args()
    res: dict = {"definition": "tasks whose tool-call sequence equals the warm start's, ignoring the report summary"}
    for sp in ("proc_conf", "proc_combo", "proc_stress"):
        w = seqs(ROOT / f"results/eval/a12/conf_reference/warm_start__neutral__{sp}.jsonl")
        res[sp] = {"n": len(w)}
        for arm in ("rv", "ra"):
            res[sp][arm] = [sum(w[t] == x[t] for t in w) for x in
                            (seqs(ROOT / f"results/eval/a12/conf/a12_proc_{arm}_s{s}__neutral__{sp}.jsonl") for s in range(5))]
    (ROOT / a.out).write_text(json.dumps(res, indent=2))
    print(json.dumps(res))


if __name__ == "__main__":
    main()
