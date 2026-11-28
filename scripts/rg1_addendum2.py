"""RG1 addendum 2 (PREREGISTRATION_RG1_ADDENDUM2.md): descriptive decompositions, written before any RG1 result was
read. CPU. Reads results/rg1/{RG,NG}_D_s{k}__rg1_test.jsonl and AB_no_retry_D_s0; writes results/rg1/rg1_addendum2.json.

1. STR, VTC and unsafe split by regime (resolvable vs unresolvable), RG vs NG (3 seeds pooled, counts).
2. Episodes whose terminal report was written by the gate (finalize or hand-off) vs by the agent.
3. NG episodes reported "failed" on unresolvable tasks that RG turned into a hand-off (relabelling).
4. VTC on unresolvable tasks (same-key resends the gate blocks), RG vs NG.
5. Reference agents behind the full gate on rg1_test (scripted, deterministic): NOOP (no call), CLAIM (reports
   completed at once), HONEST_FAIL (reports failed at once).
6. no_retry ablation: duplicate receipts (flagged bug: an unread record counts as net 0 received).
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "results/rg1"


def rows(tag):
    f = R / f"{tag}__rg1_test.jsonl"
    return [json.loads(l) for l in f.read_text().splitlines() if l.strip()] if f.exists() else None


def gate_terminal(r):
    rep = [s for s in r["trace"] if s["tool"] == "submit_report" and s["obs"].get("status") == "ok"]
    return bool(rep) and bool(rep[0].get("gate"))


def counts(rs):
    c = Counter()
    for r in rs:
        t = r["txn"]
        reg = "unresolvable" if t["unresolvable"] else "resolvable"
        c[f"{reg}:n"] += 1
        for f in ("str", "vtc", "unsafe", "escalated"):
            c[f"{reg}:{f}"] += bool(t[f])
        c["gate_terminal"] += gate_terminal(r)
    return dict(c)


def references():
    from actionproof.bench.generator_proc import DATA, load_tasks
    from actionproof.integrity.sim.runtime import episode_record_proc
    from actionproof.recovery_gate.env import make_env_gated
    from actionproof.txn import truth
    out = {}
    for name, call in (("NOOP", None), ("CLAIM", "completed"), ("HONEST_FAIL", "failed")):
        c = Counter()
        for t in load_tasks(DATA / "rg1_test.jsonl"):
            e = make_env_gated(t, f"ref-{name}", surface=False)
            if call:
                e.call("submit_report", {"status": call, "summary": "x"})
            e.gate_finalize()
            s = truth.txn_score(episode_record_proc(t, e))
            reg = "unresolvable" if s["unresolvable"] else "resolvable"
            c[f"{reg}:n"] += 1
            for f in ("str", "vtc", "unsafe", "escalated"):
                c[f"{reg}:{f}"] += bool(s[f])
        out[name] = dict(c)
    return out


def main():
    res = {}
    for arm in ("RG", "NG"):
        rs = [x for k in range(3) for x in (rows(f"{arm}_D_s{k}") or [])]
        res[arm] = counts(rs)
    relabel = Counter()
    for k in range(3):
        ng, rg = rows(f"NG_D_s{k}"), rows(f"RG_D_s{k}")
        if not ng or not rg:
            continue
        key = lambda r: (r["task_id"], r["sample"])
        rgm = {key(r): r for r in rg}
        for r in ng:
            if r["txn"]["unresolvable"] and (r["report"] or {}).get("status") == "failed":
                relabel["ng_failed_unresolvable"] += 1
                g = rgm.get(key(r))
                relabel["rg_escalated_same_episode"] += bool(g and g["txn"]["escalated"])
                relabel["rg_hand_off_by_gate"] += bool(g and gate_terminal(g))
    res["relabel"] = dict(relabel)
    ab = rows("AB_no_retry_D_s0")
    if ab:
        res["no_retry_dup_receipts"] = sum(1 for r in ab if r["txn"]["dmr"] and any(
            a["tool"] == "record_goods_receipt" for a in r["attempts"]))
    res["references_behind_gate"] = references()
    (R / "rg1_addendum2.json").write_text(json.dumps(res, indent=1, sort_keys=True) + "\n")
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
