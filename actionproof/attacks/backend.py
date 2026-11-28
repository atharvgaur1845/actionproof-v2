"""Thin adapter between the attack suite and a transaction simulator (docs/attacks/SPEC.md §6).

The suite needs four things from a backend: an env class to put AttackMixin (and optionally GateMixin) in front of,
the constructor arguments for a task, an episode record, and per-episode annotations. TXN v1 (`txn_v1`) is the only
backend today; TXN v2 (branch research/txn-v2) plugs in by registering a second `Backend` here.

Hooks AttackMixin uses on the env (TXN v1 has all of them; a v2 env must provide them or the adapter must shim them):
    call(tool, args, by) -> reply; trace (list of entries with idx/tool/args/obs/by[/txn.stale]); attempts (with by,
    tool, po_id); state (tables of row dicts); version (per service); lag[service] = {"view", "as_of", "remaining"}
    honoured by non-consistent reads; _insert(table, row); _audit(tool, po_id, key, status, by); _tick(idx);
    calls (Counter); _cur_args; reports.
"""
from __future__ import annotations

from typing import Any, Protocol

from actionproof.attacks.env import AttackMixin, PayeeArgMixin


class Backend(Protocol):
    name: str

    def env_class(self, payee_arg: bool, gated: bool) -> type: ...

    def build(self, cls: type, task: Any, rollout_id: str, seed: int | None) -> Any: ...

    def record(self, task: Any, env: Any) -> dict[str, Any]: ...

    def annotate(self, rec: dict[str, Any]) -> dict[str, Any]: ...


class TxnV1:
    name = "txn_v1"

    def __init__(self) -> None:
        from actionproof.recovery_gate.env import GateMixin
        from actionproof.txn.env import TxnEnv
        self._classes = {
            (False, False): type("AttackTxnEnv", (AttackMixin, TxnEnv), {}),
            (True, False): type("AttackPayeeTxnEnv", (AttackMixin, PayeeArgMixin, TxnEnv), {}),
            (False, True): type("GatedAttackTxnEnv", (GateMixin, AttackMixin, TxnEnv), {}),
            (True, True): type("GatedAttackPayeeTxnEnv", (GateMixin, AttackMixin, PayeeArgMixin, TxnEnv), {}),
        }

    def env_class(self, payee_arg: bool, gated: bool) -> type:
        return self._classes[(payee_arg, gated)]

    def build(self, cls: type, task: Any, rollout_id: str, seed: int | None) -> Any:
        """make_env_txn's construction (actionproof/txn/env.py), canonical surface only."""
        m = task.meta
        faults = m.get("faults") or []
        ratios = {(f["tool"], int(f["k"])): float(f["ratio"]) for f in faults if f.get("ratio") is not None}
        fl = [{k: f[k] for k in ("tool", "k", "kind")} for f in faults]
        return cls(task.initial_state, fl, rollout_id=rollout_id, seed=m["env_seed"] if seed is None else seed,
                   never_commit=m.get("never_commit"), ratios=ratios, txn=m.get("txn") or {})

    def record(self, task: Any, env: Any) -> dict[str, Any]:
        from actionproof.integrity.sim.runtime import episode_record_proc
        if getattr(env, "gate", None) is not None:
            env.gate_finalize()
        rec = episode_record_proc(task, env)
        if getattr(env, "gate", None) is not None:
            rec["gate"] = env.gate_record()
        rec["attack"] = env.attack_record()
        return rec

    def annotate(self, rec: dict[str, Any]) -> dict[str, Any]:
        from actionproof.attacks import harm
        from actionproof.txn import truth
        rec["txn"] = truth.txn_score(rec)
        rec["harm"] = harm.flags(rec)
        return rec


BACKENDS: dict[str, Backend] = {}


def get_backend(name: str = "txn_v1") -> Backend:
    if name not in BACKENDS:
        if name == "txn_v1":
            BACKENDS[name] = TxnV1()
        else:
            raise KeyError(f"unknown backend {name!r}; register it in actionproof/attacks/backend.py")
    return BACKENDS[name]


def make_env_attack(task: Any, rollout_id: str, seed: int | None = None, rules: Any = None,
                    backend: str = "txn_v1", gate_cls: Any = None) -> Any:
    """The env for one attack-suite task: attack layer always (a twin has no components, so it is transparent), the
    Recovery Gate in front when `rules` is not None (gate_cls defaults to the frozen RG1 Gate)."""
    from actionproof.recovery_gate.gate import Gate, intent_of
    b = get_backend(backend)
    payee = (task.meta.get("attack") or {}).get("schema") == "payee_arg"
    env = b.build(b.env_class(payee, rules is not None), task, rollout_id, seed)
    env.attach_attack(task)
    if rules is not None:
        env.attach_gate((gate_cls or Gate)(intent_of(task), rules, name=env.surface_names()))
    return env


__all__ = ["Backend", "TxnV1", "get_backend", "make_env_attack"]
