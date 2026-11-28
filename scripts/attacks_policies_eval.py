"""Development evaluation of the scripted policies on the attack suite (docs/attacks/SPEC.md §7).

B0 (ack-trusting), B4 (reconciliation) and OBD (obedient: follows tool-output text), each without and with the frozen
RG1 Recovery Gate (full rules), on every task of attack_dev. The held-out split is refused: no policy is ever run on
it before a registered evaluation.

    python scripts/attacks_policies_eval.py            # results/attacks/dev_policies.{jsonl,summary.json,md}
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

POLICIES = ("B0", "B4", "OBD")
CODE = ("actionproof/attacks/env.py", "actionproof/attacks/harm.py", "actionproof/attacks/policies.py",
        "actionproof/attacks/generate.py", "actionproof/attacks/templates.py", "actionproof/attacks/spec.py",
        "actionproof/attacks/metrics.py", "actionproof/attacks/backend.py", "actionproof/recovery_gate/gate.py",
        "actionproof/recovery_gate/env.py", "actionproof/txn/env.py", "actionproof/txn/policies.py")
KEEP = ("variant", "family", "template_id", "heldout_template", "vocab", "attack_success", "component_success",
        "unsafe_mutation", "false_completion", "harmful", "wrong_payee", "other_po", "wrong_amount", "duplicate_steps",
        "unrequested", "compensation_abuse", "goal_met", "vsc", "report", "escalated", "refused", "outcome",
        "correct_handling", "calls", "reads", "writes", "gate_calls", "gate_events", "gate_intervened")


def code_hashes() -> dict[str, str]:
    return {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in CODE}


def compact(t, rec) -> dict:
    a = t.meta["attack"]
    comps = a["components"] or a.get("twin_of") or []
    row = {"task_id": t.task_id, "pair_id": a["pair_id"], "policy": rec["policy"], "gated": rec["gated"],
           "workflow": t.workflow, "channel": "+".join(c["channel"] for c in comps),
           "phrasing": "+".join(str(c["phrasing"]) for c in comps),
           "target_steps": [c["target_step"] for c in comps], "ended_by": rec["ended_by"],
           "wall_ms": round(rec["wall_ms"], 4), "attack_events": len(rec["attack"]["log"]),
           "gate_rules_fired": sorted({f"{e['rule']}:{e['action']}" for e in (rec.get("gate") or {}).get("log") or []})}
    row.update({k: rec["harm"][k] for k in KEEP})
    if a.get("components"):
        row["direction"] = a["components"][0]["params"].get("direction")
    return row


def markdown(summary: dict) -> str:
    from actionproof.attacks.metrics import fmt
    L = ["# Attack suite ATK1: scripted policies on attack_dev", "",
         f"Generated {summary['generated']} by scripts/attacks_policies_eval.py from results/attacks/dev_policies.jsonl.",
         f"Split sha256 {summary['split_sha256']}. Rates are k/n = rate [95% Clopper-Pearson]; zero counts add the "
         "one-sided 95% upper bound.", "",
         "## Overall (attack episodes, all families)", "",
         "| arm | attack success | unsafe mutation | false completion | correct handling | safe refusal/escalation |",
         "|---|---|---|---|---|---|"]
    for arm, s in summary["arms"].items():
        a = s["all"]["attack"]
        L.append(f"| {arm} | {fmt(a['attack_success'])} | {fmt(a['unsafe_mutation'])} | {fmt(a['false_completion'])} "
                 f"| {fmt(a['correct_handling'])} | {fmt(a['safe_refusal_or_escalation'])} |")
    L += ["", "## Clean twins and mechanics-only controls", "",
          "| arm | twin clean success | twin false block | twin harmful | control harmful | control VSC |",
          "|---|---|---|---|---|---|"]
    for arm, s in summary["arms"].items():
        tw, ct = s["all"]["twin"], s["all"]["control"]
        L.append(f"| {arm} | {fmt(tw['clean_success'])} | {fmt(tw['false_block'])} | {fmt(tw['harmful'])} | "
                 f"{fmt(ct['harmful'])} | {fmt(ct['vsc'])} |")
    L += ["", "## Attack success by family", "", "| family | " + " | ".join(summary["arms"]) + " |",
          "|---|" + "---|" * len(summary["arms"])]
    fams = sorted(next(iter(summary["arms"].values()))["by_family"])
    for f in fams:
        cells = [fmt(s["by_family"][f]["attack"]["attack_success"]) for s in summary["arms"].values()]
        L.append(f"| {f} | " + " | ".join(cells) + " |")
    L += ["", "## Gate contrasts (paired, same tasks)", "",
          "| policy | attack success prevented | harm introduced by gate | gate-induced false block (twins) | "
          "extra calls (attack / twin) | extra reads (attack / twin) | extra wall ms (attack / twin) |",
          "|---|---|---|---|---|---|---|"]
    for p, c in summary["paired"].items():
        L.append(f"| {p} | {fmt(c['attack_success_prevented'])} | {fmt(c['harm_introduced_by_gate'])} | "
                 f"{fmt(c['gate_induced_false_block'])} | {c['extra_calls']['attack']} / {c['extra_calls']['twin']} | "
                 f"{c['extra_reads']['attack']} / {c['extra_reads']['twin']} | "
                 f"{c['extra_wall_ms']['attack']} / {c['extra_wall_ms']['twin']} |")
    L += ["", "## Cost per episode (means)", "", "| arm | variant | calls | reads | writes | gate calls | wall ms |",
          "|---|---|---|---|---|---|---|"]
    for arm, s in summary["arms"].items():
        for v, c in s["all"]["cost"].items():
            L.append(f"| {arm} | {v} | {c['calls']} | {c['reads']} | {c['writes']} | {c['gate_calls']} | "
                     f"{c['wall_ms']} |")
    return "\n".join(L) + "\n"


def main() -> None:
    from actionproof.attacks.generate import load_split, split_hash
    from actionproof.attacks.metrics import breakdown, paired, summarise
    from actionproof.attacks.policies import run_episode
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="attack_dev", choices=["attack_dev"])
    ap.add_argument("--out", default="results/attacks/dev_policies")
    a = ap.parse_args()
    out = ROOT / a.out
    tasks = load_split(a.split)
    t0 = time.time()
    rows = []
    for t in tasks:
        for p in POLICIES:
            for g in (False, True):
                rows.append(compact(t, run_episode(p, t, gate=g)))
    secs = time.time() - t0
    out.parent.mkdir(parents=True, exist_ok=True)
    out.with_suffix(".jsonl").write_text("".join(json.dumps(r, sort_keys=True, ensure_ascii=False) + "\n"
                                                 for r in rows))
    arms = {}
    for p in POLICIES:
        for g in (False, True):
            rs = [r for r in rows if r["policy"] == p and r["gated"] == g]
            arms[f"{p}{'+gate' if g else ''}"] = {"all": summarise(rs), "by_family": breakdown(rs, "family"),
                                                 "by_channel": breakdown(rs, "channel"),
                                                 "by_phrasing": breakdown(rs, "phrasing"),
                                                 "by_workflow": breakdown(rs, "workflow")}
    pair = {}
    for p in POLICIES:
        al = [r for r in rows if r["policy"] == p and not r["gated"]]
        gd = [r for r in rows if r["policy"] == p and r["gated"]]
        pair[p] = paired(al, gd)
        pair[p]["by_family"] = {f: paired([r for r in al if r["family"] == f], [r for r in gd if r["family"] == f])
                                for f in sorted({r["family"] for r in al})}
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    if not commit and (ROOT / "COMMIT").exists():  # a tar copy on the cluster carries the commit in a file
        commit = (ROOT / "COMMIT").read_text().strip()
    summary = {"generated": time.strftime("%Y-%m-%d %H:%M:%S %Z"), "split": a.split, "split_sha256": split_hash(tasks),
               "n_tasks": len(tasks), "n_episodes": len(rows), "seconds": round(secs, 2), "host": platform.node(),
               "python": platform.python_version(), "git_head": commit, "code_sha256": code_hashes(),
               "jsonl_sha256": hashlib.sha256(out.with_suffix(".jsonl").read_bytes()).hexdigest(),
               "arms": arms, "paired": pair}
    out.with_suffix(".summary.json").write_text(json.dumps(summary, indent=1, sort_keys=True, ensure_ascii=False))
    out.with_suffix(".md").write_text(markdown(summary))
    print(json.dumps({k: summary[k] for k in ("split", "n_tasks", "n_episodes", "seconds", "jsonl_sha256")}))


if __name__ == "__main__":
    main()
