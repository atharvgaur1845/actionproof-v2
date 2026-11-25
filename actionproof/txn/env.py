"""TXN benchmark environment (env_version proc3; spec: docs/txn/TXN_SPEC.md).

`TxnEnv` subclasses proc2's `S3ProcEnv` and changes no proc1/proc2 behaviour. Old fault kinds go to the frozen
write chokepoint unchanged; the new kinds (log_conflict, late_commit, concurrent, visibility_lag), the audit log, the
`consistent` read flag and the evidence regimes are implemented here. Evaluator-only data (staleness, commit times,
colleague attempts) go into the trace/attempt records, never into an observation.
"""
from __future__ import annotations

import copy
import inspect
import re
from typing import Any, Callable

from actionproof.integrity.sim.env import (FAULT_KINDS, FAULTABLE, PROC_SYSTEM, READ_SERVICE, READ_TOOLS, STEP_TOOL,
                                           TABLE_SERVICE, TABLES, TIMEOUT, TOOL_STEP, WRITE_TOOLS)
from actionproof.stage3.env import S3ProcEnv

ENV_VERSION = "proc3"
NEW_KINDS = ("log_conflict", "late_commit", "concurrent", "visibility_lag")
TXN_FAULT_KINDS = tuple(FAULT_KINDS) + NEW_KINDS + ("never_commit",)
AUDIT_TOOL = "get_audit_log"
TXN_READ_TOOLS = READ_TOOLS + (AUDIT_TOOL,)
TXN_AGENT_TOOLS = READ_TOOLS + (AUDIT_TOOL,) + WRITE_TOOLS + ("submit_report",)
MUTATIONS = ("place_supplier_order", "pay_supplier", "record_goods_receipt", "post_journal_entry", "refund_payment")
UNAVAILABLE = {"status": "error", "error": "service_unavailable"}
LATE_DELAY = 2   # a late request lands after the agent's 2nd subsequent call
LAG_READS = 2    # delayed regime: the next 2 non-consistent reads of a service see the pre-commit view
ARG_FIELD = {"pay_supplier": "amount_cents", "post_journal_entry": "amount_cents", "record_goods_receipt": "qty",
             "refund_payment": "payment_id"}
ID_FIELD = {"place_supplier_order": "order_ref", "pay_supplier": "payment_id", "record_goods_receipt": "receipt_id",
            "post_journal_entry": "entry_id", "refund_payment": "refund_id"}
CONSISTENT_DOC = ("            consistent: Optional; true reads the primary copy, which always reflects every committed "
                  "write. The default (false) may be served by a replica that can lag recent writes by a few reads.\n")


def _with_consistent(doc: str) -> str:
    return doc.rstrip() + "\n" + CONSISTENT_DOC.rstrip("\n") + "\n        "


class TxnEnv(S3ProcEnv):
    domain = "proc"
    env_version = ENV_VERSION

    def __init__(self, initial_state: dict[str, list[dict[str, Any]]], faults: list[dict[str, Any]] | None,
                 rollout_id: str, seed: int = 0, never_commit: str | None = None,
                 ratios: dict[tuple[str, int], float] | None = None, txn: dict[str, Any] | None = None) -> None:
        faults = list(faults or [])
        old = [f for f in faults if f["kind"] not in NEW_KINDS]
        super().__init__(initial_state, old, rollout_id, seed=seed, never_commit=never_commit, ratios=ratios)
        self.txn_plan = {(f["tool"], int(f["k"])): f["kind"] for f in faults if f["kind"] in NEW_KINDS}
        txn = dict(txn or {})
        self.txn_cfg = txn
        self.unavailable = frozenset(txn.get("unavailable_reads") or ())
        self.delayed = txn.get("regime") == "delayed" or "visibility_lag" in self.txn_plan.values()
        self.lag_first_reads = int(txn.get("lag_first_reads") or 0)
        self.reorder_fields = bool(txn.get("reorder_fields"))
        self.audit: list[dict[str, Any]] = []
        self.lag: dict[str, dict[str, Any]] = {}
        self.pending: list[dict[str, Any]] = []
        self.n_reads = 0
        self._cur_args: dict[str, Any] = {}
        self._read_meta: dict[str, Any] | None = None

    # ------------------------------------------------------------------ interface
    def call(self, tool: str, args: dict[str, Any] | None = None, by: str = "agent") -> dict[str, Any]:
        args = dict(args or {})
        k = self.calls[tool]
        self.calls[tool] += 1
        finished = bool(self.reports)
        self._read_meta = None
        if finished:
            obs: dict[str, Any] = {"status": "error", "error": "episode_finished"}
        elif tool not in TXN_AGENT_TOOLS:
            obs = {"status": "error", "error": "unknown_tool", "detail": tool}
        else:
            fn: Callable[..., dict[str, Any]] = getattr(self, tool)
            try:
                inspect.signature(fn).bind(**args)
            except TypeError as e:
                obs = {"status": "error", "error": "invalid_arguments", "detail": str(e)}
            else:
                if by == "agent" and self.txn_plan.get((tool, k)) == "concurrent" and tool in FAULTABLE:
                    self._colleague(tool, args)
                self._ctx = (len(self.trace), tool, k, by)
                self._cur_args = args
                try:
                    obs = fn(**args)
                except (ValueError, TypeError) as e:
                    obs = {"status": "error", "error": "invalid_arguments", "detail": str(e)}
        idx = len(self.trace)
        entry = {"idx": idx, "tool": tool, "args": args, "obs": copy.deepcopy(obs), "by": by}
        if self._read_meta is not None:
            entry["txn"] = self._read_meta
        self.trace.append(entry)
        if by == "agent" and not finished:
            self._tick(idx)
        return obs

    def tools(self) -> list[Callable[..., dict[str, Any]]]:
        def make(name: str) -> Callable[..., dict[str, Any]]:
            method = getattr(self, name)

            def tool(**kwargs: Any) -> dict[str, Any]:
                return self.call(name, kwargs)

            tool.__name__ = tool.__qualname__ = name
            tool.__doc__ = method.__doc__
            tool.__signature__ = inspect.signature(method)  # type: ignore[attr-defined]
            tool.__annotations__ = dict(method.__annotations__)
            return tool

        return [make(n) for n in TXN_AGENT_TOOLS]

    # ------------------------------------------------------------------ colleague and late commits
    def _colleague(self, tool: str, args: dict[str, Any]) -> None:
        po = self._po(args.get("po_id")) if tool != "refund_payment" else None
        if po is None:
            return
        cargs: dict[str, Any] = {"po_id": po["po_id"]}
        if tool in ("pay_supplier", "post_journal_entry"):
            cargs["amount_cents"] = po["amount_cents"]
        if tool == "record_goods_receipt":
            cargs["qty"] = po["qty"]
        else:
            cargs["idempotency_key"] = f"COL-{po['po_id']}"
        idx = len(self.trace)
        self._ctx = (idx, tool, -1, "colleague")
        self._cur_args = cargs
        saved, self.never_commit = self.never_commit, None
        try:
            obs = getattr(self, tool)(**cargs)
        finally:
            self.never_commit = saved
        self.trace.append({"idx": idx, "tool": tool, "args": cargs, "obs": copy.deepcopy(obs), "by": "colleague"})

    def _tick(self, idx: int) -> None:
        for p in list(self.pending):
            if p["created_idx"] == idx:  # the faulted call itself does not count
                continue
            p["remaining"] -= 1
            if p["remaining"] <= 0 or self.reports:
                self.pending.remove(p)
                self._land(p, idx)

    def _land(self, p: dict[str, Any], idx: int) -> None:
        at = p["attempt"]
        if p["scope"] is not None and p["scope"] in self.idem:
            at["late"] = {"landed_after_idx": idx, "deduplicated": True}
            return
        snap = self._snapshot()
        for t, r in p["rows"](p["new_id"], False):
            self._insert(t, r)
        if p["scope"] is not None:
            self.idem[p["scope"]] = {"status": "ok", p["id_field"]: p["new_id"]}
        at["committed"], at["commit_idx"] = "full", idx
        at["late"] = {"landed_after_idx": idx, "deduplicated": False}
        self._open_lag(snap)

    # ------------------------------------------------------------------ delayed regime
    def _snapshot(self) -> tuple[dict[str, list[dict[str, Any]]], dict[str, int]]:
        return {t: copy.deepcopy(self.state[t]) for t in TABLES}, dict(self.version)

    def _open_lag(self, snap: tuple[dict[str, list[dict[str, Any]]], dict[str, int]]) -> None:
        if not self.delayed:
            return
        tables, version = snap
        for s, v in version.items():
            if self.version[s] != v:
                self.lag[s] = {"view": {t: tables[t] for t in TABLES if TABLE_SERVICE[t] == s}, "as_of": v,
                               "remaining": LAG_READS}

    def _txn_view(self, service: str, k: int, tool: str, consistent: bool) -> tuple[dict[str, Any], int, bool]:
        """(view, as_of, stale) for one read call (spec §4.4)."""
        first = self.n_reads < self.lag_first_reads
        self.n_reads += 1
        if consistent:
            return self.state, self.version[service], False
        w = self.lag.get(service)
        if w and w["remaining"] > 0:
            w["remaining"] -= 1
            return {**self.state, **w["view"]}, w["as_of"], True
        if self.fault_plan.get((tool, k)) == "replica_lag" or first:
            lag = self.version[service] > 0
            return {**self.state, **self.prev[service]}, self.version[service] - lag, lag
        return self.state, self.version[service], False

    # ------------------------------------------------------------------ write chokepoint
    def _write(self, tool, po_id, key, prefix, rows, id_field):  # type: ignore[override]
        idx, _, k, by = self._ctx
        scope = (tool, po_id, key) if key is not None else None
        kind = None if by == "colleague" or (scope and scope in self.idem) else self.txn_plan.get((tool, k))
        if tool not in FAULTABLE:
            kind = None
        snap = self._snapshot()
        n_att = len(self.attempts)
        audit_status = None
        if kind in ("log_conflict", "visibility_lag", "late_commit"):
            new_id = self._next_id(prefix)
            if kind == "late_commit":
                committed = "none"
            else:
                committed = "full"
                for t, r in rows(new_id, False):
                    self._insert(t, r)
                if scope:
                    self.idem[scope] = {"status": "ok", id_field: new_id}
            obs = copy.deepcopy(TIMEOUT)
            self.attempts.append({"idx": idx, "tool": tool, "po_id": po_id, "fault": kind, "committed": committed,
                                  "from_idempotency_store": False, "obs_status": obs["status"]})
            if kind == "late_commit":
                self.pending.append({"attempt": self.attempts[-1], "scope": scope, "rows": rows, "new_id": new_id,
                                     "id_field": id_field, "remaining": LATE_DELAY, "created_idx": idx})
            audit_status = "error" if kind == "log_conflict" else "timeout"
        else:
            if by == "colleague":
                inner = rows

                def rows(i: str, part: bool):  # noqa: F811  colleague rows carry by="colleague"
                    return [(t, {**r, "by": "colleague"}) for t, r in inner(i, part)]
            obs = super()._write(tool, po_id, key, prefix, rows, id_field)
            if kind == "concurrent" and self.attempts[-1]["fault"] is None:
                self.attempts[-1]["fault"] = "concurrent"
            audit_status = "timeout" if obs.get("error") == "timeout" else obs.get("status")
        for at in self.attempts[n_att:]:
            at.update(by=by, key=key, args=copy.deepcopy(self._cur_args),
                      commit_idx=idx if at["committed"] != "none" else None)
        self._audit(tool, po_id, key, audit_status, by)
        self._open_lag(snap)
        return obs

    def _audit(self, tool: str, po_id: str, key: str | None, status: str | None, by: str) -> None:
        e: dict[str, Any] = {"request_id": f"REQ-{len(self.audit) + 1:04d}", "tool": tool, "idempotency_key": key}
        f = ARG_FIELD.get(tool)
        if f is not None:
            e[f] = self._cur_args.get(f)
        e.update(ack_status=status, by=by)
        self.audit.append({"po_id": po_id, "entry": e})

    def escalate_to_human(self, po_id: str, reason: str) -> dict[str, Any]:
        """Hand a purchase order to a human operator.

        Args:
            po_id: Purchase order id.
            reason: Why a human is needed.
        """
        n = len(self.attempts)
        obs = super().escalate_to_human(po_id, reason)
        for at in self.attempts[n:]:
            at.update(by=self._ctx[3], key=None, args=copy.deepcopy(self._cur_args), commit_idx=self._ctx[0])
        return obs

    # ------------------------------------------------------------------ reads
    def _reorder(self, obs: dict[str, Any]) -> dict[str, Any]:
        if not self.reorder_fields or obs.get("status") != "ok":
            return obs

        def rev(x: Any) -> Any:
            if isinstance(x, dict):
                return {k: rev(x[k]) for k in reversed(list(x))}
            if isinstance(x, list):
                return [rev(v) for v in x]
            return x
        return rev(obs)

    def _meta(self, consistent: bool, stale: bool, unavailable: bool, service: str | None) -> None:
        self._read_meta = {"consistent": consistent, "stale": stale, "unavailable": unavailable,
                           "primary_as_of": self.version[service] if service else len(self.audit)}

    @staticmethod
    def _flag(consistent: Any) -> bool:
        if not isinstance(consistent, bool):
            raise ValueError("consistent must be true or false")
        return consistent

    def _read(self, po_id: str, table: str, cols: tuple[str, ...], key: str, consistent: Any) -> dict[str, Any]:
        _, tool, k, _ = self._ctx
        consistent = self._flag(consistent)
        s = TABLE_SERVICE[table]
        if tool in self.unavailable:
            self._meta(consistent, False, True, s)
            return dict(UNAVAILABLE)
        if self._po(po_id) is None:
            return {"status": "error", "error": "not_found"}
        view, as_of, stale = self._txn_view(s, k, tool, consistent)
        self._meta(consistent, stale, False, s)
        out = [{**{c: r[c] for c in cols}, **({"by": r["by"]} if "by" in r else {})}
               for r in view[table] if r["po_id"] == po_id]
        return self._reorder({"status": "ok", key: out, "as_of": as_of})

    def get_purchase_order(self, po_id: str, consistent: bool = False) -> dict[str, Any]:
        _, tool, k, _ = self._ctx
        consistent = self._flag(consistent)
        if tool in self.unavailable:
            self._meta(consistent, False, True, "oms")
            return dict(UNAVAILABLE)
        if self._po(po_id) is None:
            return {"status": "error", "error": "not_found"}
        view, as_of, stale = self._txn_view("oms", k, tool, consistent)
        self._meta(consistent, stale, False, "oms")
        po = next((r for r in view["pos"] if r["po_id"] == po_id), None)
        if po is None:
            return {"status": "error", "error": "not_found"}
        sup = next((r for r in view["suppliers"] if r["supplier_id"] == po["supplier_id"]), None)
        return self._reorder({"status": "ok", "po": dict(po), "supplier": dict(sup) if sup else None, "as_of": as_of})

    def find_supplier_order(self, po_id: str, consistent: bool = False) -> dict[str, Any]:
        return self._read(po_id, "supplier_orders", ("order_ref", "sku", "qty", "status"), "orders", consistent)

    def list_payments(self, po_id: str, consistent: bool = False) -> dict[str, Any]:
        return self._read(po_id, "payments", ("payment_id", "amount_cents", "payee_account", "kind", "refund_of"),
                          "payments", consistent)

    def list_goods_receipts(self, po_id: str, consistent: bool = False) -> dict[str, Any]:
        return self._read(po_id, "receipts", ("receipt_id", "qty", "kind"), "receipts", consistent)

    def get_journal_entries(self, po_id: str, consistent: bool = False) -> dict[str, Any]:
        return self._read(po_id, "journal", ("entry_id", "amount_cents", "kind"), "entries", consistent)

    def get_audit_log(self, po_id: str) -> dict[str, Any]:
        """List the write requests the services received for a purchase order, oldest first, with the acknowledgement
        each request was sent (ok, error or timeout). The log records requests, not their effects.

        Args:
            po_id: Purchase order id.
        """
        if AUDIT_TOOL in self.unavailable:
            self._meta(False, False, True, None)
            return dict(UNAVAILABLE)
        if self._po(po_id) is None:
            return {"status": "error", "error": "not_found"}
        self._meta(False, False, False, None)
        reqs = [copy.deepcopy(a["entry"]) for a in self.audit if a["po_id"] == po_id]
        return self._reorder({"status": "ok", "requests": reqs, "as_of": len(reqs)})


# proc1 docstrings + the documented `consistent` parameter (signature order matches)
from actionproof.integrity.sim.env import ProcEnv as _P  # noqa: E402

for _name in READ_TOOLS:
    getattr(TxnEnv, _name).__doc__ = _with_consistent(getattr(_P, _name).__doc__ or "")


# ---------------------------------------------------------------------- renamed surfaces (OOD alias / schema groups)
ALIAS_TOOLS = {
    "get_purchase_order": "open_purchase_order", "find_supplier_order": "supplier_order_inquiry",
    "list_payments": "payment_ledger_view", "list_goods_receipts": "goods_receipt_register",
    "get_journal_entries": "gl_journal_view", "get_audit_log": "request_audit_trail",
    "place_supplier_order": "dispatch_purchase_order", "pay_supplier": "execute_vendor_payment",
    "record_goods_receipt": "book_goods_arrival", "post_journal_entry": "commit_gl_posting",
    "refund_payment": "reclaim_payment", "escalate_to_human": "hand_to_reviewer", "submit_report": "file_final_report",
}
SCHEMA_ARGS = {"po_id": "purchase_ref", "amount_cents": "amount_in_cents", "idempotency_key": "dedup_token",
               "qty": "unit_count", "payment_id": "payment_ref_id", "reason": "escalation_note",
               "status": "report_status", "summary": "report_summary", "consistent": "read_primary"}
SURFACES = {"alias": (ALIAS_TOOLS, {}), "schema": ({}, SCHEMA_ARGS)}
KEEP_KEYS = ("status", "reason", "summary")  # obs keys never renamed (as in SR1): ack status, row status


def held_out_names() -> set[str]:
    """Every name used by SR1's training surfaces, the S4 held-out surface and the canonical API."""
    from actionproof.stage4 import shift, surface as SR1
    names = set(shift.TOOL_MAP) | set(shift.ARG_MAP.values()) | set(shift.ARG_MAP)
    names |= {"result", "ACCEPTED", "REJECTED", "deadline_exceeded"}
    for pool in list(SR1.TOOL_SYN.values()) + list(SR1.ARG_SYN.values()):
        names |= set(pool)
    for tup in SR1.ACK_SYN:
        names |= set(tup)
    names |= set(TXN_AGENT_TOOLS) | set(SR1.TOOL_SYN) | set(SR1.ARG_SYN) | {"consistent"}
    return names


def _check_surfaces() -> None:
    bad = held_out_names()
    for name, (tmap, amap) in SURFACES.items():
        new = set(tmap.values()) | set(amap.values())
        assert not new & bad, (name, new & bad)
        assert len(set(tmap.values())) == len(tmap) and len(set(amap.values())) == len(amap), name


_check_surfaces()


class TxnSurfaceEnv(TxnEnv):
    """TxnEnv behind renamed tools (alias) or renamed arguments (schema). The trace keeps canonical names."""

    def __init__(self, *a: Any, surface: str, **kw: Any) -> None:
        super().__init__(*a, **kw)
        tmap, amap = SURFACES[surface]
        self.surface = surface
        self.c2t = {c: tmap.get(c, c) for c in TXN_AGENT_TOOLS}
        self.t2c = {v: c for c, v in self.c2t.items()}
        self.arg = dict(amap)
        self.inv_arg = {v: c for c, v in amap.items()}

    def _rename(self, x: Any) -> Any:
        if isinstance(x, list):
            return [self._rename(v) for v in x]
        if isinstance(x, dict):
            out = {}
            for k, v in x.items():
                nk = k if k in KEEP_KEYS else self.arg.get(k, k)
                out[nk] = self.c2t.get(v, v) if k == "tool" and isinstance(v, str) else self._rename(v)
            return out
        return x

    def adapt_prompt(self, base: str) -> str:
        text = base if base.startswith(PROC_SYSTEM) else PROC_SYSTEM
        for c, new in sorted(self.c2t.items(), key=lambda kv: -len(kv[0])):
            text = re.sub(rf"\b{c}\b", new, text)
        if "status" in self.arg:
            text = text.replace('status "completed", "failed" or "escalated"',
                                f'{self.arg["status"]} "completed", "failed" or "escalated"')
        return text

    def call(self, tool: str, args: dict[str, Any] | None = None, by: str = "agent") -> dict[str, Any]:
        if by != "agent":
            return super().call(tool, args, by)
        if tool not in self.t2c:
            obs = {"status": "error", "error": "unknown_tool", "detail": tool}
            self.calls[tool] += 1
            idx = len(self.trace)
            self.trace.append({"idx": idx, "tool": tool, "args": dict(args or {}), "obs": obs, "by": by})
            if not self.reports:
                self._tick(idx)
            return self._rename(obs)
        canon = {self.inv_arg.get(k, k): v for k, v in (args or {}).items()}
        return self._rename(super().call(self.t2c[tool], canon, by))

    def tools(self) -> list[Callable[..., dict[str, Any]]]:
        out = []
        for c in TXN_AGENT_TOOLS:
            new, method = self.c2t[c], getattr(self, c)
            sig = inspect.signature(method)
            params = [p.replace(name=self.arg.get(p.name, p.name)) for p in sig.parameters.values()]
            doc = method.__doc__ or ""
            for a, b in self.arg.items():
                doc = re.sub(rf"\b{a}\b", b, doc)
            for a, b in self.c2t.items():
                doc = re.sub(rf"\b{a}\b", b, doc)

            def tool(_new=new, **kwargs: Any) -> dict[str, Any]:
                return self.call(_new, kwargs)
            tool.__name__ = tool.__qualname__ = new
            tool.__doc__ = doc
            tool.__signature__ = sig.replace(parameters=params)  # type: ignore[attr-defined]
            tool.__annotations__ = {self.arg.get(k, k): v for k, v in method.__annotations__.items()}
            out.append(tool)
        return out

    def to_surface(self, tool: str, args: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        return self.c2t[tool], {self.arg.get(k, k): v for k, v in args.items()}


# ---------------------------------------------------------------------- factory
def make_env_txn(task: Any, rollout_id: str, seed: int | None = None, surface: bool = True) -> TxnEnv:
    """Build the proc3 env for a task (same construction as make_env_s3, plus meta.txn). `surface=False` forces the
    canonical names (scripted policies run on the canonical surface)."""
    m = task.meta
    faults = m.get("faults") or []
    ratios = {(f["tool"], int(f["k"])): float(f["ratio"]) for f in faults if f.get("ratio") is not None}
    txn = m.get("txn") or {}
    kw = dict(rollout_id=rollout_id, seed=m["env_seed"] if seed is None else seed,
              never_commit=m.get("never_commit"), ratios=ratios, txn=txn)
    fl = [{k: f[k] for k in ("tool", "k", "kind")} for f in faults]
    if surface and txn.get("surface"):
        return TxnSurfaceEnv(task.initial_state, fl, surface=txn["surface"], **kw)
    return TxnEnv(task.initial_state, fl, **kw)


__all__ = ["TxnEnv", "TxnSurfaceEnv", "make_env_txn", "NEW_KINDS", "TXN_AGENT_TOOLS", "TXN_READ_TOOLS", "AUDIT_TOOL",
           "MUTATIONS", "UNAVAILABLE", "ENV_VERSION", "READ_SERVICE", "STEP_TOOL", "TOOL_STEP", "ALIAS_TOOLS",
           "SCHEMA_ARGS", "held_out_names"]
