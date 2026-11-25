"""R7 (exploratory; written 2026-11-26 after the 2x2 was known, so nothing here is confirmatory).

What does the RS-GRPO step add on top of recovery SFT (Dw -> D, +0.140 sampled VSC), and how does D still fail?
Reads only the frozen-test episode files (results/p1/test/{Dw,D_s*,A_s*}__*.jsonl, results/p2/ref/W__*.jsonl).

1. VSC by fault class, split and start type for W, A, Dw, D, with the paired task bootstrap of D - Dw.
2. Behaviour per arm (sampled episodes): reads, redundant reads, refunds, new-key share on re-pay, false success,
   unrepaired duplicates, honest failure, calls per success, how episodes end.
3. Task-level transitions Dw -> D: tasks whose mean sampled VSC rose / fell by at least 0.5.
4. D's failure taxonomy (sampled episodes with vsc = 0): false success by the first non-verified step verdict,
   honest failure, no tool call, context end, collateral; per split.

    python scripts/audit/r7_grpo_increment.py      # -> results/audit/r7_grpo_increment.json
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
TEST = ("p1_test_full", "p1_test_pp", "p1_test_rec", "p1_test_wf4", "p1_test_combo")
ARMS = {"W": ("results/p2/ref", ["W"]), "A": ("results/p1/test", [f"A_s{s}" for s in range(3)]),
        "Dw": ("results/p1/test", ["Dw"]), "D": ("results/p1/test", [f"D_s{s}" for s in range(3)])}


def episodes(arm: str) -> list[dict]:
    d, tags = ARMS[arm]
    out = []
    for tag in tags:
        for sp in TEST:
            f = ROOT / d / f"{tag}__{sp}.jsonl"
            for r in map(json.loads, f.read_text().splitlines()):
                r["_split"], r["_tag"] = sp, tag
                out.append(r)
    return out


def task_vsc(eps: list[dict]) -> dict:
    by = defaultdict(list)
    for r in eps:
        if r["sample"] >= 0:
            by[(r["_split"], r["task_id"])].append(r["vsc"])
    return {k: float(np.mean(v)) for k, v in by.items()}


def boot(d: np.ndarray, rng, B=10_000) -> dict:
    bs = d[rng.integers(0, len(d), size=(B, len(d)))].mean(1)
    return {"delta": float(d.mean()), "ci95": [float(np.quantile(bs, .025)), float(np.quantile(bs, .975))], "n": int(len(d))}


def behaviour(eps: list[dict]) -> dict:
    s = [r for r in eps if r["sample"] >= 0]
    n = len(s)
    succ = [r for r in s if r["vsc"]]
    rep = sum(r.get("repay_after_refund", 0) for r in s)
    return {"n": n, "vsc": sum(r["vsc"] for r in s) / n,
            "false_success": sum(r["score"]["false_success"] for r in s) / n,
            "unrepaired_duplicate": sum(bool(r.get("violation")) for r in s) / n,
            "honest_fail": sum(bool(r.get("honest_fail")) for r in s) / n,
            "reads_per_episode": sum(r.get("policy_reads", 0) for r in s) / n,
            "redundant_reads_per_episode": sum(r.get("redundant_reads", 0) for r in s) / n,
            "refunds_per_episode": sum(r.get("refunds", 0) for r in s) / n,
            "new_key_share_on_repay": (sum(r.get("repay_new_key", 0) for r in s) / rep) if rep else None,
            "calls_per_success": (sum(r.get("policy_calls", 0) for r in succ) / len(succ)) if succ else None,
            "ended_by": dict(Counter(r.get("ended_by") for r in s))}


def failure_kind(r: dict) -> str:
    sc = r["score"]
    if r.get("ended_by") == "context":
        return "context_end"
    if r.get("ended_by") == "no_call":
        return "no_tool_call"
    if sc.get("collateral"):
        return "collateral_write"
    if sc["false_success"]:
        bad = [f"{st}:{v}" for st, v in sc["steps"].items() if v != "VERIFIED_SUCCESS"]
        return "false_success:" + (bad[0] if bad else "unknown")
    if r.get("honest_fail"):
        return "honest_fail_or_escalation"
    return "other:" + str((r.get("report") or {}).get("status"))


def main() -> None:
    rng = np.random.default_rng(20261126)
    E = {a: episodes(a) for a in ARMS}
    T = {}
    for a, eps in E.items():  # arm task score = mean over its seed models of their task means
        per_model = defaultdict(list)
        for r in eps:
            per_model[r["_tag"]].append(r)
        tv = [task_vsc(v) for v in per_model.values()]
        keys = sorted(set.intersection(*(set(x) for x in tv)))
        T[a] = {k: float(np.mean([x[k] for x in tv])) for k in keys}
    meta = {}
    for r in E["Dw"]:
        meta[(r["_split"], r["task_id"])] = (r.get("fault_class") or "none", r.get("start_type") or "full")
    keys = sorted(set.intersection(*(set(t) for t in T.values())))
    out = {"status": "EXPLORATORY (R7; after the 2x2 was known)", "n_tasks": len(keys), "arms": {}, "by": {}}
    for a in ARMS:
        out["arms"][a] = behaviour(E[a])
    for name, keyf in (("split", lambda k: k[0]), ("fault_class", lambda k: meta[k][0].split(":")[0] if ":" in meta[k][0] else meta[k][0]),
                       ("start_type", lambda k: meta[k][1])):
        groups = defaultdict(list)
        for k in keys:
            groups[keyf(k)].append(k)
        out["by"][name] = {g: {"n": len(ks), **{a: float(np.mean([T[a][k] for k in ks])) for a in ARMS},
                               "D_minus_Dw": boot(np.array([T["D"][k] - T["Dw"][k] for k in ks]), rng)}
                           for g, ks in sorted(groups.items())}
    d = np.array([T["D"][k] - T["Dw"][k] for k in keys])
    out["transitions_Dw_to_D"] = {"improved_ge_0.5": int((d >= 0.5).sum()), "worsened_ge_0.5": int((d <= -0.5).sum()),
                                  "unchanged_within_0.5": int((abs(d) < 0.5).sum()), "overall": boot(d, rng)}
    fails = [r for r in E["D"] if r["sample"] >= 0 and not r["vsc"]]
    out["D_failure_taxonomy"] = {"n_failed_sampled_episodes": len(fails),
                                 "overall": dict(Counter(failure_kind(r) for r in fails).most_common()),
                                 "by_split": {sp: dict(Counter(failure_kind(r) for r in fails if r["_split"] == sp).most_common(6))
                                              for sp in TEST}}
    dst = ROOT / "results/audit/r7_grpo_increment.json"
    dst.write_text(json.dumps(out, indent=1))
    print(json.dumps({"transitions": out["transitions_Dw_to_D"], "D_fail_top": dict(list(out["D_failure_taxonomy"]["overall"].items())[:6])}, indent=1))


if __name__ == "__main__":
    main()
