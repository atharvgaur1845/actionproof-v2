"""ID1 (PREREGISTRATION_ID1): the no-verification warm start and R_V from it.

Per model: phantom read-back, VSC and false success on proc_conf; the warm start's sampled gate next to the A12 warm
start's; read rate in training rollouts per block of 50 steps (n_verify > 0); and the registered reading.
    python scripts/id1_stats.py
"""
from __future__ import annotations

import gzip
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "results/id1"


def summ(p: Path) -> dict | None:
    if not p.exists():
        return None
    s = json.loads(p.read_text())
    return {k: s[k] for k in ("n", "vsc", "false_success", "phantom_readback", "benign_readback", "mean_calls")}


def read_rate(tag: str) -> dict | None:
    p = R / "rollouts" / f"{tag}.jsonl.gz"
    if not p.exists():
        return None
    by: dict = {}
    with gzip.open(p, "rt") as f:
        for line in f:
            r = json.loads(line)
            b = f"steps_{int(r['step']) // 50 * 50}"
            n, k = by.get(b, (0, 0))
            by[b] = (n + 1, k + (int(r["n_verify"]) > 0))
    return {b: k / n for b, (n, k) in sorted(by.items())}


def main() -> None:
    out: dict = {"warm_v0": {"proc_conf": summ(R / "warm_v0__neutral__proc_conf.summary.json"),
                             "gate_T1x4": summ(R / "warm_v0__neutral__proc_train_T1x4.summary.json")},
                 "a12_warm_gate_T1x4": summ(ROOT / "results/eval/a12/init__neutral__proc_train_T1x4.summary.json")}
    rb = []
    for s in (0, 1):
        tag = f"id1_rv_s{s}"
        c = summ(R / f"{tag}__neutral__proc_conf.summary.json")
        out[tag] = {"proc_conf": c, "train_read_rate": read_rate(tag),
                    "dev": {st: summ(R / "dev" / f"{tag}_step{st}__neutral__proc_dev.summary.json") for st in (100, 200)}}
        if c:
            rb.append(c["phantom_readback"]["rate"] or 0.0)
    if len(rb) == 2:
        out["reading"] = ("selection" if max(rb) <= 0.10 else "discovery" if min(rb) >= 0.50 else "mixed")
    (R / "id1_summary.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
