"""World-independent advantage scale for AD-D2 (docs/research/algorithm_discovery/EXPERIMENT_PROTOCOL_D2.md).

GRPO divides each group's centred returns by that group's own std. When the group shares one hidden world, this
weights worlds by 1/sigma_w (FINAL_ALGORITHM.md P1, P4). D2 keeps the group-mean centring and divides every
advantage at step t by one scalar: an exponential moving average of the mean within-group std over steps < t. The
divisor therefore does not depend on the current step's worlds. It keeps GRPO's typical step size; 'none' scaling
is smaller by a factor of about 1/sigma.

  s_t = max(floor, ema_{t-1}),   ema_t = d * ema_{t-1} + (1 - d) * m_t,   m_t = mean_g std(centred A in group g)
  step 0: ema_{-1} = m_0 (the only step whose divisor uses its own batch; logged).
"""
from __future__ import annotations

import json
from pathlib import Path

DECAY = 0.9
FLOOR = 0.02


class EmaScale:
    def __init__(self, decay: float = DECAY, floor: float = FLOOR, ema: float | None = None) -> None:
        self.decay, self.floor, self.ema = decay, floor, ema

    def divisor_and_update(self, group_stds: list[float]) -> tuple[float, float]:
        """Returns (divisor used for this step, batch statistic m_t); then updates the EMA with m_t."""
        m = sum(group_stds) / len(group_stds)
        prev = m if self.ema is None else self.ema
        div = max(self.floor, prev)
        self.ema = prev * self.decay + (1 - self.decay) * m
        return div, m


def group_stds(adv: list[float], G: int) -> list[float]:
    """Population std within each contiguous block of G centred advantages."""
    out = []
    for i in range(0, len(adv), G):
        g = adv[i:i + G]
        mu = sum(g) / len(g)
        out.append((sum((x - mu) ** 2 for x in g) / len(g)) ** 0.5)
    return out


def install(trainer_cls, log_path: Path, G: int = 8) -> EmaScale:
    """Wrap `_generate_and_score_completions`: TRL runs with scale_rewards='none' (centred returns); this divides the
    advantages by the EMA divisor. Resumes the EMA from the last logged value if the log exists."""
    import torch
    ema = None
    if log_path.exists():
        rows = [json.loads(l) for l in log_path.read_text().splitlines() if l.strip()]
        if rows:
            ema = rows[-1]["ema_after"]
    sc = EmaScale(ema=ema)
    orig = trainer_cls._generate_and_score_completions

    def wrapped(self, inputs):
        out = orig(self, inputs)
        a = out["advantages"].float()
        flat = a.tolist() if a.dim() == 1 else a[:, 0].tolist()
        div, m = sc.divisor_and_update(group_stds(flat, G))
        out["advantages"] = (out["advantages"] / div).to(out["advantages"].dtype)
        with log_path.open("a") as f:
            f.write(json.dumps({"step": int(self.state.global_step), "divisor": div, "batch_mean_group_std": m,
                                "ema_after": sc.ema}) + "\n")
        return out

    trainer_cls._generate_and_score_completions = wrapped
    return sc
