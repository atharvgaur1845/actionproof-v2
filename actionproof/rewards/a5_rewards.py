"""Amendment 5 reward arms. They differ ONLY in the information source of the task-success term.

    R_src = T_src + LAMBDA * K,   T_src = 1[success_src] - MU * 1[duplicate_src],   src in {state, ack}

T is the SAME functional form in both arms; only where success and duplication are read from differs:
  state : success = goal predicates hold at the hidden horizon ∧ exactly one committed refund ∧ no policy
          violation ∧ no collateral write;  duplicate = more than one committed refund on the order
          (environment state + operation registry).
  ack   : success = the last ack of every required write was "success";  duplicate = more than one SUCCESS ack
          for a refund on the order (what the acks say). No violation term: acks carry no policy information, and
          a receipt-based I would leak state (a refund after a committed-but-unacked one is policy-invalid because
          the order is already refunded). This leak exists in Amendment 4's registered ack_outcome (H10) and is
          logged there as a confound; here it is excluded by design.
MU = 1: without a duplicate penalty, the horizon truth pays a blind retry +1 in the never-commits world of an
observation-equivalent pair, so blind retry would tie with honest abstention in expectation
(tests/test_a5_rewards.py::test_expected_ordering_on_unresolvable_pair).
K       : IDENTICAL in both arms and TRANSCRIPT-ONLY. Is the report + certificate consistent with what the agent
          itself observed from authoritative evidence tools? It never reads hidden state, so it adds no
          state information to the ack arm. It supervises certificate/report form identically, so a
          difference between arms cannot come from proof-specific shaping.
LAMBDA  : 0.5, fixed a priori (not tuned). It makes a consistent report on an unresolvable task (K = 1, T = luck)
          beat an inconsistent "completed" claim in expectation. There was no budget for a sensitivity
          analysis; this is reported as a limitation.
"""
from __future__ import annotations

from typing import Any

from actionproof.env.tools_a5 import A5_EVIDENCE_TOOLS

LAMBDA = 0.5
MU = 1.0


def t_state(rec: dict[str, Any]) -> float:
    v = rec["a5v"]
    return float(v["goal_final"] and not v["duplicate"] and not v["collateral"]) - MU * float(v["duplicate"])


def t_ack(rec: dict[str, Any]) -> float:
    trace, goal = rec["trace"], rec["goal"]
    for op, entity in goal["required_writes"]:
        last = [s["obs"] for s in trace if s["tool"] == op and isinstance(s.get("args"), dict)
                and entity in s["args"].values() and not s.get("by")]
        if not last or not isinstance(last[-1], dict) or last[-1].get("status") != "success":
            ok = False
            break
    else:
        ok = True
    order = next((e for op, e in goal["required_writes"] if op == "issue_refund"), None)
    acks = [s for s in trace if s["tool"] == "issue_refund" and not s.get("by") and isinstance(s.get("obs"), dict)
            and s["obs"].get("status") == "success" and isinstance(s.get("args"), dict) and s["args"].get("order_id") == order]
    return float(ok) - MU * float(len({s["obs"].get("op_id") for s in acks}) > 1)


def _observed_ops(steps: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Latest authoritative view of each operation, as the agent saw it (get_operation / list_operations)."""
    seen: dict[str, dict[str, Any]] = {}
    for s in steps:
        obs = s.get("obs") if isinstance(s.get("obs"), dict) else {}
        views = [obs.get("operation")] if s["tool"] == "get_operation" else obs.get("operations", []) \
            if s["tool"] == "list_operations" else []
        for v in views or []:
            if isinstance(v, dict) and v.get("op_id"):
                seen[v["op_id"]] = {**v, "_now": obs.get("now")}
    return seen


def _evidences_committed(step: dict[str, Any], op_id: str, refund_id: str | None) -> bool:
    obs = step.get("obs") if isinstance(step.get("obs"), dict) else {}
    if step["tool"] == "get_operation":
        o = obs.get("operation") or {}
        return o.get("op_id") == op_id and o.get("status") == "committed"
    if step["tool"] == "list_operations":
        return any(o.get("op_id") == op_id and o.get("status") == "committed" for o in obs.get("operations", []))
    if step["tool"] == "query_ledger" and refund_id:
        return any(r.get("refund_id") == refund_id for r in obs.get("refunds", []))
    return False


def consistency(trace: list[dict[str, Any]], report: dict[str, Any] | None) -> float:
    """K: 1 iff the status and certificate are supported by the agent's own observations (transcript-only)."""
    if not report:
        return 0.0
    n_r = report.get("trace_len", len(trace))
    steps = [s for s in trace[:n_r] if not s.get("by")]
    idx = {i for i, s in enumerate(trace[:n_r]) if not s.get("by")}
    cert = report.get("certificate") if isinstance(report.get("certificate"), dict) else {}
    effects = [e for e in (cert.get("effects") if isinstance(cert.get("effects"), list) else []) if isinstance(e, dict)]
    evidence = [i for i in (cert.get("evidence") if isinstance(cert.get("evidence"), list) else []) if isinstance(i, int)]
    unresolved = [x for x in (cert.get("unresolved") if isinstance(cert.get("unresolved"), list) else []) if isinstance(x, str)]
    if any(i not in idx or trace[i]["tool"] not in A5_EVIDENCE_TOOLS for i in evidence):
        return 0.0
    seen = _observed_ops(steps)
    own = []
    for s in steps:
        if s["tool"] == "issue_refund" and isinstance(s.get("obs"), dict):
            x = s["obs"].get("op_id") or s["obs"].get("request_id")
            if x and x not in own:
                own.append(x)
    proven = []
    for e in effects:
        v = seen.get(e.get("op_id"))
        if v is None or v.get("actor") != e.get("actor") or v.get("status") != e.get("state"):
            return 0.0
        if e.get("state") == "committed":
            if not any(_evidences_committed(trace[i], v["op_id"], v.get("refund_id")) for i in evidence):
                return 0.0
            proven.append(v)
    own_latest = {o: seen.get(o, {}).get("status") for o in own}  # None = never checked after the write
    pending_unbounded = [o for o, st in own_latest.items() if st == "pending" and (
        seen[o].get("settle_bound_s") is None or (seen[o]["_now"] or 0) - seen[o]["accepted_at"] <= seen[o]["settle_bound_s"])]
    st = report.get("status")
    if st == "completed":
        ok = bool(proven) and all(s_ != "pending" for s_ in own_latest.values())
    elif st == "failed":
        ok = bool(own) and all(s_ == "failed" for s_ in own_latest.values()) and not any(
            v.get("status") == "committed" and v.get("tool") == "issue_refund" for v in seen.values())
    elif st in ("uncertain", "escalated"):
        ok = bool(pending_unbounded) and set(pending_unbounded) <= set(unresolved)
    else:
        ok = False
    return float(ok and set(unresolved) <= set(pending_unbounded))


def a5_reward(rec: dict[str, Any], source: str) -> float:
    t = t_state(rec) if source == "state" else t_ack(rec)
    return t + LAMBDA * consistency(rec["trace"], rec["report"])
