"""Where does an acknowledgement-grounded reward select against reading back a refund? (CPU; Amendment 8 audit.)

Input: warm-start samples at step 0 (T=1 x k per pool-B task, the gate files). Each episode is rescored under the
arm's reward; within each task group GRPO's advantage is (r - mean) / std (zero-variance groups give 0). For the
marker `reads`, i.e. a model-issued verification read of the refunded order after the first `issue_refund` call and
before the report (and, second, the H7-3 template marker `demo_key`), we report per task class:

  * mixed groups (groups containing both readers and non-readers),
  * sum and mean of the within-group advantage differential adv(readers) - adv(non-readers),
  * mean reward of readers and of non-readers.

The sum is the step-0 selection on reading; negative = the first gradient step pushes reading down. The classes the
Amendment 8 run (X1) removes from training are `refund timeout` (RP, issue_refund, F4_timeout: the refund committed
and the ack says error) and `refund UR` (UR, issue_refund: the refund never commits). The `X1 pool` row is the
prediction for the X1 task list: the sum over the classes it keeps.

    python scripts/a8_ack_class_audit.py results/raw/a6/gate_init__a4_train_B_T1x4.jsonl \
        results/raw/a7/gate_init__a4_train_B_T1x4.jsonl > results/a8/ack_class_audit.md
"""
from __future__ import annotations

import json
import statistics
import sys
from collections import defaultdict

from actionproof.bench.generator_a4 import load_tasks
from actionproof.env.tools_ecom import VERIFICATION_TOOLS
from actionproof.eval.a4_metrics import _ids, model_steps
from actionproof.rewards import a6_rewards as A6
from actionproof.rewards.ack_outcome import ack_outcome_score

POOL = {t.task_id: t for t in load_tasks("data/bench/a4/a4_train_B.jsonl")}
REMOVED = ("refund timeout", "refund UR")


def task_class(task_id: str) -> str:
    a4 = POOL[task_id].meta["a4"]
    if a4["fault_tool"] == "issue_refund" and a4["env_class"] == "UR":
        return "refund UR"
    if a4["fault_tool"] == "issue_refund" and a4["fault_type"] == "F4_timeout":
        return "refund timeout"
    if a4["fault_tool"] == "issue_refund":
        return "refund phantom"
    return "other (OK, F5, audit/close faults)"


def reads(rec: dict) -> bool | None:
    """True/False if the episode called issue_refund; None otherwise."""
    t = model_steps(rec["trace"])
    first = next((i for i, s in enumerate(t) if s["tool"] == "issue_refund"), None)
    if first is None:
        return None
    ids = _ids(t[first])
    for s in t[first + 1:]:
        if s["tool"] == "submit_report":
            break
        if s["tool"] in VERIFICATION_TOOLS and _ids(s) & ids:
            return True
    return False


def demo_key(rec: dict) -> bool | None:
    """True/False if the episode called issue_refund: first refund keyed `refund-<order>` (the H7-3 marker)."""
    t = model_steps(rec["trace"])
    ref = next((s for s in t if s["tool"] == "issue_refund"), None)
    if ref is None:
        return None
    a = ref.get("args") if isinstance(ref.get("args"), dict) else {}
    return a.get("idempotency_key") == f"refund-{a.get('order_id')}"


MARKERS = {"reads": reads, "demo_key": demo_key}


def _trace(r):
    return [s for s in r["trace"] if isinstance(s.get("args"), dict)]


ARMS = {
    "M4 ack_outcome": lambda r: ack_outcome_score(POOL[r["task_id"]].goal, _trace(r), r["verdict"]["I"]),
    "M3 outcome": lambda r: float(r["verdict"]["P"] == 1 and r["verdict"]["I"] == 0),
    "M2 noreport": lambda r: A6.pcstr_d_noreport_score(__import__("types").SimpleNamespace(**r["verdict"])),
}


def audit(path: str, arm: str, marker: str = "reads") -> list[str]:
    fn, mk = ARMS[arm], MARKERS[marker]
    groups = defaultdict(list)
    for line in open(path):
        r = json.loads(line)
        groups[r["task_id"]].append(r)
    per = defaultdict(lambda: {"diffs": [], "rr": [], "rn": []})
    for tid, g in groups.items():
        lab = [mk(r) for r in g]
        if any(x is None for x in lab) or all(lab) or not any(lab):
            continue
        rs = [fn(r) for r in g]
        c = per[task_class(tid)]
        c["rr"] += [x for x, l in zip(rs, lab) if l]
        c["rn"] += [x for x, l in zip(rs, lab) if not l]
        sd = statistics.pstdev(rs)
        if sd < 1e-9:
            c["diffs"].append(0.0)
            continue
        adv = [(x - statistics.mean(rs)) / sd for x in rs]
        c["diffs"].append(statistics.mean([a for a, l in zip(adv, lab) if l]) -
                          statistics.mean([a for a, l in zip(adv, lab) if not l]))
    out = [f"**{arm}, marker `{marker}`**", "", "| task class | mixed groups | sum adv(marked) − adv(unmarked) |"
           " mean per group | reward, marked | reward, unmarked |", "|---|---|---|---|---|---|"]
    tot = kept = 0.0
    n_tot = n_kept = 0
    for k in ("refund timeout", "refund UR", "refund phantom", "other (OK, F5, audit/close faults)"):
        c = per.get(k)
        if not c or not c["diffs"]:
            out.append(f"| {k} | 0 | – | – | – | – |")
            continue
        s, n = sum(c["diffs"]), len(c["diffs"])
        tot, n_tot = tot + s, n_tot + n
        if k not in REMOVED:
            kept, n_kept = kept + s, n_kept + n
        out.append(f"| {k} | {n} | {s:+.2f} | {s / n:+.3f} | {statistics.mean(c['rr']):.2f} (n={len(c['rr'])}) | "
                   f"{statistics.mean(c['rn']):.2f} (n={len(c['rn'])}) |")
    out.append(f"| **full pool** | {n_tot} | **{tot:+.2f}** | {tot / max(n_tot, 1):+.3f} | | |")
    out.append(f"| **X1 pool** (refund timeout and refund UR removed) | {n_kept} | **{kept:+.2f}** | "
               f"{kept / max(n_kept, 1):+.3f} | | |")
    return out + [""]


def main() -> None:
    lines = ["# Step-0 selection on reading back a refund, by task class (Amendment 8 audit, CPU)", "",
             "Generated by `scripts/a8_ack_class_audit.py`. Marker: a model-issued verification read of the refunded "
             "order after the first `issue_refund` call. Caveats: the gate files were sampled in the evaluation "
             "schema (D-EVAL-1), groups of 4 (training uses 8), 100 tasks; step-0 only, not training dynamics.", ""]
    for p in sys.argv[1:]:
        lines += [f"## `{p}`", ""]
        for marker in MARKERS:
            for arm in ARMS:
                lines += audit(p, arm, marker)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
