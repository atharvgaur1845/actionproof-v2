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


def pcstr_d_reward(prompts, completions, **kwargs) -> list[float]:
    """PCSTR with the disclosure fix: silence on failure costs as much as a false claim."""
    return [pcstr_score(_verdict(e), disclosure=True) for e in _episodes(completions, kwargs)]


def make_pcstr_c_reward(cost: float = 0.02) -> Callable[..., list[float]]:
    def pcstr_c_reward(prompts, completions, **kwargs) -> list[float]:
        return [pcstr_score(_verdict(e), e["n_verify"], cost, disclosure=True)  # PCSTR-d + cost (prereg)
                for e in _episodes(completions, kwargs)]
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


def make_ack_outcome_reward(goals: dict[str, GoalSpec]) -> Callable[..., list[float]]:
    """Amendment 4 H10: `outcome` with the goal term read from acks instead of the DB (rewards/ack_outcome.py)."""
    from actionproof.rewards.ack_outcome import ack_outcome_score

    def ack_outcome_reward(prompts, completions, **kwargs) -> list[float]:
        return [ack_outcome_score(goals[e["task_id"]], e["trace"], e["verdict"]["I"])
                for e in _episodes(completions, kwargs)]
    return ack_outcome_reward


def make_a5_reward(source: str) -> Callable[..., list[float]]:
    """Amendment 5: T_source + LAMBDA * K (K identical across arms; rewards/a5_rewards.py)."""
    from actionproof.rewards.a5_rewards import a5_reward

    def a5r(prompts, completions, **kwargs) -> list[float]:
        return [a5_reward(e, source) for e in _episodes(completions, kwargs)]
    a5r.__name__ = f"a5_{source}_reward"
    return a5r


def a5_proof_valid_metric(prompts, completions, **kwargs) -> list[float]:
    return [float(e["a5v"]["proof_valid"]) for e in _episodes(completions, kwargs)]


def _a6(fn: Callable[[Verdict], float], name: str) -> Callable[..., list[float]]:
    """Amendment 6 arms (rewards/a6_rewards.py): same episode record, different scoring of the verdict."""
    def r(prompts, completions, **kwargs) -> list[float]:
        return [fn(_verdict(e)) for e in _episodes(completions, kwargs)]
    r.__name__ = f"{name}_reward"
    return r


def make_a6_reward(arm: str) -> Callable[..., list[float]]:
    from actionproof.rewards import a6_rewards as A
    fn = {"pcstr_d_noreport": A.pcstr_d_noreport_score, "pcstr_d_noND": A.pcstr_d_noND_score,
          "outcome_G": A.outcome_G_score}[arm]
    return _a6(fn, arm)
