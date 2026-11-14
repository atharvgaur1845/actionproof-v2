"""Build demo/data.json: replayed episodes (step cards) and chart data, every number with its source file (CPU).

    python scripts/build_demo_data.py
"""
from __future__ import annotations

import json
import math
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "demo"))
from episode_html import build_lane  # noqa: E402
from render import score  # noqa: E402

from actionproof.bench.generator import make_split  # noqa: E402

MODELS = {  # display name -> (channel, val episodes)
    "Warm start (RFT v2)": ("warm", "results/eval/r1/rft_tiny_v2__neutral__val.jsonl"),
    "GRPO + tool-trusting reward": ("ack", "results/eval/r3/grpo_tiny_v2_tool_trust_s0__neutral__val.jsonl"),
    "GRPO + acknowledgement reward": ("ack", "results/eval/r3/grpo_tiny_v2_ack_outcome_s0__neutral__val.jsonl"),
    "GRPO + PCSTR-d (database reward)": ("state", "results/eval/r3/grpo_tiny_v2_pcstr_d_s0__neutral__val.jsonl"),
}
PER_FAULT = 6
CONF = "results/confirmatory.json"
ARM_LABEL = {"rft_v2": ("Before RL (warm start)", "warm"), "tool_trust": ("Trusts the tool", "ack"),
             "llm_judge": ("AI judge reads transcript", "ack"), "outcome": ("Reads database (pass/fail)", "state"),
             "pcstr": ("Reads database (PCSTR)", "state"), "pcstr_c": ("Reads database (PCSTR-c)", "state"),
             "pcstr_d": ("Reads database (PCSTR-d, ours)", "state")}


def wilson(p: float, n: int) -> list[float]:
    z = 1.96
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(c - h, 4), round(c + h, 4)]


def episodes() -> tuple[list, list]:
    runs = {n: {json.loads(line)["task_id"]: json.loads(line) for line in open(ROOT / p)} for n, (_, p) in MODELS.items()}
    common = sorted(set.intersection(*(set(r) for r in runs.values())))
    tasks = {t.task_id: t for t in make_split("val")}
    by_fault, per_fault = defaultdict(list), defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
    for tid in common:
        t = tasks[tid]
        if t.meta.get("domain") == "bank":
            continue
        fault = "+".join(t.meta["faults"]) or "benign"
        by_fault[fault].append((len({runs[n][tid]["vsc"] for n in runs}) == 1, tid))
        for n in runs:  # chart C6: outcome per fault per model over ALL common tasks
            v = runs[n][tid]["verdict"]
            k = "verified" if runs[n][tid]["vsc"] else "false_completion" if v.get("F_over") else \
                "honest_failure" if (runs[n][tid].get("report") or {}).get("status") in ("failed", "escalated") else "other"
            per_fault[n][fault][k] += 1
    eps = []
    for fault, lst in sorted(by_fault.items()):
        for _, tid in sorted(lst)[:PER_FAULT]:
            t = tasks[tid]
            eps.append({"task_id": tid, "fault": fault, "prompt": t.prompt,
                        "lanes": {n: build_lane(t, runs[n][tid], score) for n in runs}})
    c6 = [{"model": n, "fault": f, **{k: d[k] for k in ("verified", "false_completion", "honest_failure", "other")},
           "n": sum(d.values()), "source": MODELS[n][1]} for n in runs for f, d in sorted(per_fault[n].items())]
    return eps, c6


def charts() -> dict:
    conf = json.loads((ROOT / CONF).read_text())
    n = conf["n_tasks"]
    c1 = [{"arm": a, "label": ARM_LABEL[a][0], "channel": ARM_LABEL[a][1], "vsc": v, "ci95_seed0": wilson(v[0], n),
           "n": n, "seeds": len(v), "source": CONF} for a, v in conf["per_arm_vsc"].items() if a in ARM_LABEL]
    c2 = [r for r in json.loads((ROOT / "docs/figures/figures_data.json").read_text()) if r.get("figure") == "F1"]
    c4 = []
    for f, names in (("aya8b", {"aya8b__llm_judge": "Aya 8B, our transcript judge", "aya8b__auditkit_tc": "Aya 8B, AuditKIT TaskCompletion"}),
                     ("aya32b", {"aya32b__auditkit_tc": "Aya Expanse 32B, AuditKIT TaskCompletion", "aya32b__llm_judge": "Aya Expanse 32B, our transcript judge"})):
        d = json.loads((ROOT / f"results/judge/{f}__summary.json").read_text())
        for k, label in names.items():
            if k in d:
                a, vis = d[k]["ALL|FAR"], d[k].get("ALL|visible=True|FAR", {})
                c4.append({"judge": label, "far": a["rate"], "ci95": a.get("ci95"), "n": a["n"], "far_visible": vis.get("rate"),
                           "source": f"results/judge/{f}__summary.json:{k}"})
    c5 = []
    for line in (ROOT / "results/agentic_metrics.md").read_text().splitlines():
        m = re.match(r"\| (\S+) \| (\d+) \| ([\d.]+) \| ([\d.]+) \| ([\d.]+) \| ([\d.]+) \| ([\d.]+) \| ([\d.]+) \| ([\d,]+) \|", line)
        if m:
            name = m.group(1)
            fam = "warm" if name.startswith("rft") else "state" if re.search(r"outcome_s|pcstr|noreport", name) and "ack" not in name else "ack"
            c5.append({"model": name, "family": fam, "vsc": float(m.group(3)), "acc": float(m.group(4)),
                       "sec": float(m.group(8)), "chars": int(m.group(9).replace(",", "")), "source": "results/agentic_metrics.md"})
    c3 = [  # read-back after a refund ack, greedy test_iid; counts computed 2026-11-14 from the cluster eval files
        {"cell": "All training tasks (control)", "reads": 0, "n": 169, "vsc": 0.615},
        {"cell": "60 random tasks removed (placebo)", "reads": 0, "n": 163, "vsc": 0.615},
        {"cell": "2 refund task types removed (X1)", "reads": 103, "n": 154, "vsc": 0.755}]
    for r in c3:
        r.update(rate=round(r["reads"] / r["n"], 3), seeds=1,
                 source="results/eval/a6/{a7_ack_outcome,a8x1p_ack_outcome,a8x1_ack_outcome}_s0__neutral__test_iid.jsonl "
                        "(HF dataset actionproof-traces); docs/ARTICLE_DRAFT.md §3")
    c3_diff = {"diff": 0.669, "ci95": [0.591, 0.740], "n_pairs": 154, "source": "docs/ARTICLE_DRAFT.md §3 (paired task bootstrap, 10k)"}
    c3_step0 = [{"cls": c, "readers": a, "non_readers": b, "source": "results/a8/ack_class_audit.md"} for c, a, b in
                (("refund timeout", 0.08, 0.62), ("refund unrecoverable", 0.25, 1.00), ("refund phantom", 1.00, 0.87),
                 ("all other tasks", 0.71, 0.76))]
    lb = json.loads((ROOT / "results/leaderboard.json").read_text())
    c7 = [{"model": r["model"], "prompt": r["prompt"], "vsc": r["VSC"]["p"], "ci95": [r["VSC"]["lo"], r["VSC"]["hi"]],
           "n": r["n"], "source": "results/leaderboard.json"} for r in lb if r.get("prompt") == "neutral"]
    kpis = [
        {"value": "0.64 → 0.84", "label": "Verified-success completion, warm start to PCSTR-d, on 400 untouched tasks, both seeds", "source": CONF},
        {"value": "+20.25 pp", "label": "Over a tool-trusting reward, 95% CI 16.4 to 24.3", "source": CONF + ":pcstr_d_vs_tool_trust"},
        {"value": "0.72 vs 0.10", "label": "Refund read-back after a success acknowledgement, database reward vs acknowledgement reward (warm start 0.25, sampled)", "source": "docs/figures/figures_data.json (F1)"},
        {"value": "95%", "label": "Of false completions accepted by AuditKIT TaskCompletion on Aya Expanse 32B", "source": "results/judge/aya32b__summary.json"},
    ]
    return {"kpis": kpis, "c1": c1, "c1_contrasts": {"H1": conf["H1"], "vs_tool_trust": conf["pcstr_d_vs_tool_trust"]},
            "c2": c2, "c3": c3, "c3_diff": c3_diff, "c3_step0": c3_step0, "c4": c4, "c5": c5, "c7": c7}


def main() -> None:
    eps, c6 = episodes()
    ch = charts()
    ch["c6"] = c6
    git = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, cwd=ROOT).stdout.strip()
    out = {"meta": {"git": git, "split": "val", "models": list(MODELS), "channel": {n: c for n, (c, _) in MODELS.items()}},
           "charts": ch, "episodes": eps}
    p = ROOT / "demo/data.json"
    p.write_text(json.dumps(out, ensure_ascii=False, separators=(",", ":"), default=str))
    print(json.dumps({"episodes": len(eps), "bytes": p.stat().st_size, "faults": sorted({x["fault"] for x in eps})}))


if __name__ == "__main__":
    main()
