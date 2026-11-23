"""Post-hoc diagnosis of the dropped C5 candidate (exploratory; not registered). Recomputes C5's turn credit from the
recorded training rollouts (results/s5/rollouts/<tag>.jsonl.gz: per-turn progress u_k and the trace), with C5's
alpha = 0.5 and its group normalisation over all turns of the prompt's rollouts, and reports the mean normalised
local credit of read, write and report turns per 50-step block, plus reads per episode.
Episodes whose agent-call count differs from the number of progress entries (e.g. scripted recovery prefixes) are
excluded and counted. Writes results/s5/c5_diagnosis.json.
    python scripts/c5_diagnose.py
"""
from __future__ import annotations

import gzip
import json
import statistics as st
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
READ = ("get_", "list_", "find_")


def run(tag: str) -> dict:
    eps = [json.loads(line) for line in gzip.open(ROOT / f"results/s5/rollouts/{tag}.jsonl.gz", "rt")]
    groups = defaultdict(list)
    for e in eps:
        groups[(e["step"], e["task_id"])].append(e)
    by: dict = defaultdict(lambda: defaultdict(list))
    reads: dict = defaultdict(list)
    excluded = 0
    for (step, _), G in groups.items():
        rows = []
        for e in G:
            us = e["turn_progress"]
            calls = [t for t in e["trace"] if t.get("by", "agent") == "agent"]
            if len(calls) != len(us):
                excluded += 1
                continue
            acc, ls = 0.0, [0.0] * len(us)
            for k in range(len(us) - 1, -1, -1):
                acc = us[k] + 0.5 * acc
                ls[k] = acc
            rows += list(zip((c["tool"] for c in calls), ls))
            reads[int(step) // 50].append(sum(c["tool"].startswith(READ) for c in calls))
        if len(rows) < 2:
            continue
        mu, sd = st.mean(x for _, x in rows), st.pstdev([x for _, x in rows]) + 1e-4
        for tool, x in rows:
            kind = "read" if tool.startswith(READ) else ("report" if tool == "submit_report" else "write")
            by[int(step) // 50][kind].append((x - mu) / sd)
    return {"episodes": len(eps), "excluded_misaligned": excluded,
            "blocks": {f"{b * 50}-{b * 50 + 49}": {"credit": {k: {"mean": st.mean(v), "n": len(v)} for k, v in by[b].items()},
                                                   "reads_per_episode": st.mean(reads[b])} for b in sorted(by)}}


def main() -> None:
    out = {tag: run(tag) for tag in ("s5_c5_s0", "s5_c5_s1")}
    (ROOT / "results/s5/c5_diagnosis.json").write_text(json.dumps(out, indent=2))
    print(json.dumps({t: {b: {k: round(v["mean"], 3) for k, v in x["credit"].items()} for b, x in d["blocks"].items()}
                      for t, d in out.items()}))


if __name__ == "__main__":
    main()
