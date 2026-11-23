"""Pure functions for the Stage 5 development candidates (docs/research/algorithm/C5G_DEV.md). CPU-tested.

C5  verified-progress turn credit. Phi(s) = 1 - cost(s) / (2 n), with n requested steps and per-step cost from the
    verifier's own step verdict on the authoritative state (training only; the policy never sees it):
    VERIFIED 0, ABSENT 1, PARTIAL 2 for pay (refund, then re-pay) else 1, DUPLICATE 1, WRONG 2.
    u_k = Phi(after turn k) - Phi(before turn k); l_k = sum_{j>=k} alpha^(j-k) u_j;
    token advantage on turn k's policy tokens = A_episode + omega * (l_k - mean_group(l)) / (std_group(l) + eps),
    where the mean and std run over every policy turn of every rollout in the prompt group.
G   realizable hint: after a `list_payments` result whose live payments sum to more than 0 and less than the PO total
    stated in the request, one reconciliation line computed from those visible values.
"""
from __future__ import annotations

import json
import re
from statistics import mean, pstdev
from typing import Any

from actionproof.integrity.sim.truth import ABSENT, DUPLICATE, PARTIAL, VERIFIED, WRONG, _task_po, step_verdict

EPS = 1e-4
COST = {VERIFIED: 0, ABSENT: 1, DUPLICATE: 1, WRONG: 2}


def phi(task: dict[str, Any], state: dict[str, list[dict[str, Any]]]) -> float:
    po, payee = _task_po(task)
    steps = task["meta"]["requested"]
    cost = 0
    for s in steps:
        v = step_verdict(s, state, po, payee)
        cost += (2 if s == "pay" else 1) if v == PARTIAL else COST[v]
    return 1.0 - cost / (2 * len(steps))


def c5_advantages(base: list[float], meta: list[dict[str, Any]], T: int, omega: float = 1.0,
                  alpha: float = 0.5) -> tuple[list[list[float]], list[dict]]:
    """meta[i] = {"group": str, "turns": [(start, end, u), ...]}; base[i] = TRL's sequence advantage."""
    adv = [[b] * T for b in base]
    local: list[list[float]] = []
    for m in meta:
        us = [u for _, _, u in m["turns"]]
        ls, acc = [0.0] * len(us), 0.0
        for k in range(len(us) - 1, -1, -1):
            acc = us[k] + alpha * acc
            ls[k] = acc
        local.append(ls)
    groups: dict[str, list[int]] = {}
    for i, m in enumerate(meta):
        groups.setdefault(m["group"], []).append(i)
    diag = []
    for g, idx in groups.items():
        vals = [l for i in idx for l in local[i]]
        if not vals:
            continue
        mu, sd = mean(vals), pstdev(vals) + EPS
        for i in idx:
            for (s, e, _), l in zip(meta[i]["turns"], local[i]):
                a = base[i] + omega * (l - mu) / sd
                for t in range(s, min(e, T)):
                    adv[i][t] = a
        diag.append({"group": g, "mu": mu, "sd": sd, "n_turns": len(vals), "any_progress": any(v != 0 for v in vals)})
    return adv, diag


def recon_line(obs: dict[str, Any], total: int) -> str | None:
    """The G hint for one `list_payments` observation, from visible values only, or None."""
    pays = obs.get("payments") if isinstance(obs, dict) else None
    if not isinstance(pays, list):
        return None
    refunded = {p.get("refund_of") for p in pays if p.get("kind") == "refund"}
    live = sum(p.get("amount_cents", 0) for p in pays if p.get("kind") == "payment" and p.get("payment_id") not in refunded)
    if not 0 < live < total:
        return None
    return (f"Reconciliation (computed from the records above): purchase order total {total} cents; live payments "
            f"{live} cents; outstanding {total - live} cents.")


def request_total(prompt: str) -> int | None:
    """The PO total as stated in the request text ("total 174705 cents")."""
    m = re.search(r"total (\d+) cents", prompt)
    return int(m.group(1)) if m else None


def hinted_result(text: str, line: str) -> str:
    tail = "\nContinue the task."
    assert text.count(tail) == 1
    return text.replace(tail, "\n" + line + tail)


def _demo() -> None:  # pragma: no cover
    print(json.dumps({"phi_empty_3step": 1 - 3 / 6}))
