"""Amendment 12 rewards for ProcureSim (spec: docs/stage2/VERIFIER_SPEC.md §6), as TRL reward adapters.

Both read the same per-sample episode record (`ap_episodes`) and differ only in the information channel:
R_V reads the final state (via `score`); R_A reads only the acknowledgements the agent received.
"""
from __future__ import annotations

from typing import Any


def _eps(completions: list[Any], kwargs: dict[str, Any]) -> list[dict[str, Any]]:
    eps = kwargs.get("ap_episodes")
    if eps is None or len(eps) != len(completions):
        raise RuntimeError("ap_episodes missing or misaligned; is the ProcureSim rollout_func active?")
    return eps


def proc_rv_reward(prompts, completions, **kwargs) -> list[float]:
    return [float(e["score"]["r_v"]) for e in _eps(completions, kwargs)]


def proc_ra_reward(prompts, completions, **kwargs) -> list[float]:
    return [float(e["score"]["r_a"]) for e in _eps(completions, kwargs)]


def proc_vsc_metric(prompts, completions, **kwargs) -> list[float]:
    """Logged with weight 0 in both arms, so each arm reports true verified completion."""
    return [float(e["score"]["vsc"]) for e in _eps(completions, kwargs)]


def proc_phantom_readback_metric(prompts, completions, **kwargs) -> list[float]:
    """Per-episode phantom read-back share (0 when the episode had no phantom attempt); weight 0, for logging."""
    out = []
    for e in _eps(completions, kwargs):
        n, d = e["score"]["phantom_readback"]
        out.append(n / d if d else 0.0)
    return out


ARMS = {"proc_rv": proc_rv_reward, "proc_ra": proc_ra_reward}
