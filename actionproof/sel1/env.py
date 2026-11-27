"""SEL1 env: the frozen proc3 `TxnEnv` plus one regime, a transient read outage.

`meta.txn.transient_unavailable = {tool: n}`: the first n agent calls of `tool` return
`{"status": "error", "error": "service_unavailable"}` (the TXN unavailable reply, byte-identical), and every later
call is served normally. The tool must also be listed in `meta.txn.unavailable_reads`, so that the first call
behaves exactly as in TXN's unresolvable regime. Calls by a colleague never consume the outage. Everything else is
TxnEnv's behaviour, unchanged (`actionproof/txn/env.py` is frozen by `scripts/freeze_txn1.py`).
"""
from __future__ import annotations

from typing import Any

from actionproof.txn.env import TxnEnv


class SelEnv(TxnEnv):
    def __init__(self, *a: Any, **kw: Any) -> None:
        super().__init__(*a, **kw)
        tr = dict(self.txn_cfg.get("transient_unavailable") or {})
        for tool in tr:
            assert tool in self.unavailable, f"transient tool {tool} must also be in unavailable_reads"
        self.transient_left = {t: int(n) for t, n in tr.items()}

    def call(self, tool: str, args: dict[str, Any] | None = None, by: str = "agent") -> dict[str, Any]:
        if by == "agent" and tool in self.transient_left and not self.reports:
            if self.transient_left[tool] > 0:
                self.transient_left[tool] -= 1
                self.unavailable = self.unavailable | {tool}
            else:
                self.unavailable = self.unavailable - {tool}
        return super().call(tool, args, by)


def make_env_sel(task: Any, rollout_id: str, seed: int | None = None) -> TxnEnv:
    """`make_env_txn` with SelEnv for tasks that carry a transient outage; canonical tool names only (SEL1 uses no
    renamed surface)."""
    from actionproof.txn.env import make_env_txn
    m = task.meta
    txn = m.get("txn") or {}
    assert not txn.get("surface"), "SEL1 tasks use the canonical surface"
    if not txn.get("transient_unavailable"):
        return make_env_txn(task, rollout_id, seed=seed, surface=False)
    faults = m.get("faults") or []
    ratios = {(f["tool"], int(f["k"])): float(f["ratio"]) for f in faults if f.get("ratio") is not None}
    fl = [{k: f[k] for k in ("tool", "k", "kind")} for f in faults]
    return SelEnv(task.initial_state, fl, rollout_id=rollout_id, seed=m["env_seed"] if seed is None else seed,
                  never_commit=m.get("never_commit"), ratios=ratios, txn=txn)
