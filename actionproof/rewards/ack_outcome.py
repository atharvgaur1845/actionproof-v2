"""`ack_outcome` (PREREGISTRATION_A4 H10): the source-only ablation of `outcome`.

    outcome      = 1[P = 1 ∧ I = 0]                              goal satisfaction read from the DB
    ack_outcome  = 1[∀(op, e) ∈ W: last ack of (op, e) is success] · 1[I = 0]   ... read from the acks

Same functional form (binary, report-blind, same invariant term); only the information source of the goal term
differs. I is computed from receipts in both. It does not depend on fault injection, with one exception: a
phantom write produces no receipt, so a policy-violating phantom write is invisible to I. Its frequency is
reported alongside H10.
"""
from __future__ import annotations

from typing import Any

from actionproof.env.verifier import GoalSpec


def acked_all(spec: GoalSpec, trace: list[dict[str, Any]]) -> bool:
    for op, entity in spec.required_writes:
        last = [s["obs"] for s in trace if s["tool"] == op and isinstance(s.get("args"), dict)
                and entity in s["args"].values()]
        if not last or not isinstance(last[-1], dict) or last[-1].get("status") != "success":
            return False
    return True


def ack_outcome_score(spec: GoalSpec, trace: list[dict[str, Any]], I: float) -> float:
    return float(acked_all(spec, trace) and I == 0)
