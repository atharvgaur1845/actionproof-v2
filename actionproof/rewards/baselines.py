"""Baseline rewards compared against PCSTR (spec §11).

`tool_trust` models what many pipelines do implicitly: it believes the tool's own
acknowledgements and the agent's final claim. `outcome` is the strong state baseline:
a binary DB check without receipts, partial credit or honesty terms.
"""
from __future__ import annotations

from typing import Any

from actionproof.env.verifier import GoalSpec, Verdict


def tool_trust_score(spec: GoalSpec, trace: list[dict[str, Any]], report: dict[str, str] | None) -> float:
    """(share of required writes whose LAST observation was a success ack) x 1[claimed completed]."""
    if not spec.required_writes:
        acked = 1.0
    else:
        hits = 0
        for op, entity in spec.required_writes:
            last = [s["obs"] for s in trace if s["tool"] == op and entity in s["args"].values()]
            hits += bool(last) and last[-1].get("status") == "success"
        acked = hits / len(spec.required_writes)
    claimed = (report or {}).get("status") == spec.expected_status
    return acked * float(claimed)


def outcome_score(v: Verdict) -> float:
    """1 iff every postcondition and invariant holds in the final DB state."""
    return float(all(v.details["postconditions_met"]) and v.I == 0)
