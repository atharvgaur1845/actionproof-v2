"""Cached demo replays from REAL evaluation episodes (no GPU at demo time).

Each recorded episode is replayed deterministically (env/counterfactual.replay asserts every
observation matches) and rendered with the demo's claimed-vs-actual panel. Tasks are those every
chosen model was evaluated on; per fault type, tasks where the models DISAGREE on VSC come first.

    python scripts/build_replays.py --out demo/replays/index.json --per-fault 6 \
        "RFT baseline=results/eval/r1/rft_tiny_curated__neutral__val.jsonl" \
        "GRPO + tool_trust=results/eval/r3/grpo_tiny_tool_trust_s0__neutral__val.jsonl"
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "demo"))
from render import render, score  # noqa: E402

from actionproof.bench.generator import make_split  # noqa: E402
from actionproof.env.counterfactual import replay  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("models", nargs="+", help="'Display name=episodes.jsonl' (same split for all)")
    ap.add_argument("--out", default="demo/replays/index.json")
    ap.add_argument("--per-fault", type=int, default=6)
    a = ap.parse_args()
    runs = {}
    for m in a.models:
        name, path = m.split("=", 1)
        runs[name] = {json.loads(l)["task_id"]: json.loads(l) for l in open(path)}
    common = sorted(set.intersection(*(set(r) for r in runs.values())))
    split = common[0].rsplit("_", 1)[0]
    tasks = {t.task_id: t for t in make_split(split)}
    by_fault: dict[str, list] = {}
    for tid in common:
        t = tasks[tid]
        if t.meta.get("domain") == "bank":
            continue
        fault = "+".join(t.meta["faults"]) or "benign"
        disagree = len({runs[n][tid]["vsc"] for n in runs}) > 1
        by_fault.setdefault(fault, []).append((not disagree, tid))
    items = []
    for fault, lst in sorted(by_fault.items()):
        for _, tid in sorted(lst)[: a.per_fault]:
            t = tasks[tid]
            panels = {}
            for name, recs in runs.items():
                env = replay(t, recs[tid])
                panels[name] = render(t, env, score(t, env))
            items.append({"fault": fault, "task_id": tid, "prompt": t.prompt, "panels": panels})
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps({"models": list(runs), "split": split, "items": items}, ensure_ascii=False))
    print(json.dumps({"items": len(items), "faults": {f: sum(i["fault"] == f for i in items) for f in by_fault}}))


if __name__ == "__main__":
    main()
