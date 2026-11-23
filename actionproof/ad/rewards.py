"""Rewards for the costed-verification testbed. Every estimator optimises the same objective,
J = E[R_V - c * reads], where reads counts the agent's calls to read tools (get_purchase_order, find_supplier_order,
list_payments, list_goods_receipts, get_journal_entries) and R_V is ActionProof's report-blind state reward.

  rc     R_V - c * reads                          (GRPO, no-std and batch-std arms)
  awrb   E[R_V | trajectory] - c * reads          (aliased-world Rao-Blackwellisation; same expectation as rc)
  cvar   CVaR_alpha[R_V | trajectory] - c * reads (risk-sensitive variant; a different objective)

The episode record carries `ad` = actionproof.ad.worlds.marginalise(...) computed in the rollout.
"""
from __future__ import annotations

from typing import Any

COST = 0.05


def _eps(completions: list[Any], kw: dict[str, Any]) -> list[dict[str, Any]]:
    eps = kw.get("ap_episodes")
    if eps is None or len(eps) != len(completions):
        raise RuntimeError("ap_episodes missing or misaligned")
    return eps


def base_value(e: dict[str, Any], field: str) -> float:
    if field == "r_v":
        return float(e["score"]["r_v"])
    v = (e.get("ad") or {}).get(field)
    if v is None:
        raise RuntimeError(f"episode {e.get('rollout_id')} has no {field}: world outside the prior")
    return float(v)


def make_reward(field: str, cost: float = COST):
    def f(prompts, completions, **kw) -> list[float]:
        return [base_value(e, field) - cost * e["n_verify"] for e in _eps(completions, kw)]
    f.__name__ = {"r_v": "rc", "r_bar": "awrb", "cvar": "cvar"}[field] + "_reward"
    return f


def metric(name: str):
    def f(prompts, completions, **kw) -> list[float]:
        out = []
        for e in _eps(completions, kw):
            ad = e.get("ad") or {}
            out.append(float({"vsc": e["score"]["vsc"], "r_v": e["score"]["r_v"], "reads": e["n_verify"],
                              "r_bar": ad.get("r_bar") or 0.0, "post_fault": ad.get("post_fault") or 0.0,
                              "dup": e["score"]["duplicate_irreversible"]}[name]))
        return out
    f.__name__ = f"m_{name}"
    return f


METRICS = [metric(n) for n in ("vsc", "r_v", "reads", "r_bar", "post_fault", "dup")]
ESTIMATORS = {  # estimator -> (reward field, TRL scale_rewards)
    "grpo": ("r_v", "group"),
    "nostd": ("r_v", "none"),
    "batchstd": ("r_v", "batch"),
    "awrb": ("r_bar", "group"),
    "awrbn": ("r_bar", "none"),
    "cvar": ("cvar", "group"),
}
