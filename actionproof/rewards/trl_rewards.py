"""TRL/AgentTune reward-function adapters: `fn(prompts, completions, **kwargs) -> list[float]`.

Every arm reads the same per-sample `ap_episodes` extra field produced by our rollout_func,
so arms differ ONLY in how they score an identical episode record (spec §13 shared-init design).
"""
from __future__ import annotations

from typing import Any, Callable

from actionproof.env.verifier import GoalSpec, Verdict
from actionproof.rewards.baselines import tool_trust_score
from actionproof.rewards.pcstr import pcstr_score


def _episodes(completions: list[Any], kwargs: dict[str, Any]) -> list[dict[str, Any]]:
    eps = kwargs.get("ap_episodes")
    if eps is None or len(eps) != len(completions):
        raise RuntimeError(f"ap_episodes missing or misaligned: {None if eps is None else len(eps)} "
                           f"vs {len(completions)} completions; is the ActionProof rollout_func active?")
    return eps


def _verdict(e: dict[str, Any]) -> Verdict:
    return Verdict(**e["verdict"], details={"violations": e["violations"]})


def pcstr_reward(prompts, completions, **kwargs) -> list[float]:
    return [pcstr_score(_verdict(e)) for e in _episodes(completions, kwargs)]


def make_pcstr_c_reward(cost: float = 0.02) -> Callable[..., list[float]]:
    def pcstr_c_reward(prompts, completions, **kwargs) -> list[float]:
        return [pcstr_score(_verdict(e), e["n_verify"], cost) for e in _episodes(completions, kwargs)]
    return pcstr_c_reward


def outcome_reward(prompts, completions, **kwargs) -> list[float]:
    eps = _episodes(completions, kwargs)
    return [float(e["verdict"]["P"] == 1.0 and e["verdict"]["I"] == 0) for e in eps]


def make_tool_trust_reward(goals: dict[str, GoalSpec]) -> Callable[..., list[float]]:
    def tool_trust_reward(prompts, completions, **kwargs) -> list[float]:
        return [tool_trust_score(goals[e["task_id"]], e["trace"], e["report"])
                for e in _episodes(completions, kwargs)]
    return tool_trust_reward


def vsc_metric(prompts, completions, **kwargs) -> list[float]:
    """Logged alongside the training reward (weight 0) so every arm reports true success."""
    return [float(e["vsc"]) for e in _episodes(completions, kwargs)]
