"""Softmax decision policy for the tabular testbed. One linear head per decision type over observable features.

Feature modes:
  tab   [1, onehot(op type), kf]                                   (a per-type lookup table: can memorise)
  feat  [1, logit p, log s, log c, kf, logit a, log(1+D), log(V+Lf)]  (observable risk descriptors: can generalise
        to unseen types, costs and fault rates; the VoI rule p*s*(V+Lf) > c is linear in these features)
The policy observes the op type's *descriptors* (as an LLM observes an operation's name and context), never the
hidden fault plan. Action 0's logit is fixed at 0.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np

from actionproof.ad2.tabular.env import DTYPES

# warm-start action probabilities per decision type (verify prob at V1 is set by rho0)
WARM = {"R": [0.1, 0.9], "V2": [0.7, 0.3], "V2c": [0.7, 0.3], "T": [0.2, 0.3, 0.5], "TR": [0.1, 0.9],
        "TRa": [0.7, 0.3]}


def _lg(x: float) -> float:
    x = min(max(x, 1e-3), 1 - 1e-3)
    return math.log(x / (1 - x))


class Policy:
    def __init__(self, n_types: int, mode: str = "tab", rho0: float = 0.3):
        self.mode, self.n_types = mode, n_types
        self.nf = 2 + n_types if mode == "tab" else 8
        self.theta = {d: np.zeros((n, self.nf)) for d, n in DTYPES.items()}
        init = {**WARM, "V1": [1 - rho0, rho0]}
        for d, pr in init.items():
            lp = np.log(np.array(pr))
            self.theta[d][:, 0] = lp - lp[0]  # bias column

    def feats(self, ctx: dict[str, Any]) -> np.ndarray:
        o, cfg = ctx["op"], ctx["cfg"]
        if self.mode == "tab":
            x = np.zeros(self.nf)
            x[0] = 1.0
            x[1 + o.tid] = 1.0
            x[-1] = float(ctx["kf"])
            return x
        return np.array([1.0, _lg(o.p) / 4, math.log(max(o.s, 1e-3)) / 4, math.log(cfg.c + 1e-3) / 4, float(ctx["kf"]),
                         _lg(o.a) / 4, math.log1p(cfg.D), math.log(cfg.V + cfg.Lf)])

    def probs_phi(self, dtype: str, phi: np.ndarray) -> np.ndarray:
        z = self.theta[dtype] @ phi
        z = z - z.max()
        e = np.exp(z)
        return e / e.sum()

    def probs(self, dtype: str, ctx: dict[str, Any]) -> np.ndarray:
        return self.probs_phi(dtype, self.feats(ctx))

    def sample(self, dtype: str, ctx: dict[str, Any], state: Any, rng: np.random.Generator):
        phi = self.feats(ctx)
        pr = self.probs_phi(dtype, phi)
        a = int(rng.choice(len(pr), p=pr))
        rec = {"d": dtype, "phi": phi, "a": a, "pr": pr, "j": ctx["j"], "kf": ctx["kf"], "tid": ctx["op"].tid,
               "info": informative(dtype, ctx["j"], state), "state": state.copy() if dtype in ("V1", "T") else None}
        return a, rec

    def grad_logp(self, rec: dict[str, Any]) -> tuple[str, np.ndarray]:
        e = np.zeros(len(rec["pr"]))
        e[rec["a"]] = 1.0
        return rec["d"], np.outer(e - rec["pr"], rec["phi"])

    def grad_allaction(self, rec: dict[str, Any], q: np.ndarray) -> tuple[str, np.ndarray]:
        """Gradient of sum_a pi(a|h) Q(h,a) w.r.t. the head: pi_k (Q_k - Qbar) phi."""
        pr = rec["pr"]
        return rec["d"], np.outer(pr * (q - pr @ q), rec["phi"])


def informative(dtype: str, j: int, st: Any) -> bool:
    """Whether the *verify* action at this decision could change the posterior over alive rows with assumed mass."""
    m = st.alive & (st.rows.w > 0)
    Z = st.rows.Z[:, j]
    if dtype == "V1":
        obs = (Z == 1) | ((Z == 0) & (st.rows.FA[:, j] == 1))
    elif dtype in ("V2", "V2c"):
        obs = st.correct[:, j]
    elif dtype == "T":
        obs = Z == 2
    else:
        return False
    return len(np.unique(obs[m])) > 1


class Adam:
    def __init__(self, theta: dict[str, np.ndarray], lr: float, b1: float = 0.9, b2: float = 0.999, eps: float = 1e-8):
        self.lr, self.b1, self.b2, self.eps, self.t = lr, b1, b2, eps, 0
        self.m = {k: np.zeros_like(v) for k, v in theta.items()}
        self.v = {k: np.zeros_like(v) for k, v in theta.items()}

    def step(self, theta: dict[str, np.ndarray], grad: dict[str, np.ndarray]) -> None:
        """Gradient *ascent*. Row 0 of every head stays fixed."""
        self.t += 1
        for k, g in grad.items():
            g = g.copy()
            g[0] = 0.0
            self.m[k] = self.b1 * self.m[k] + (1 - self.b1) * g
            self.v[k] = self.b2 * self.v[k] + (1 - self.b2) * g * g
            mh = self.m[k] / (1 - self.b1 ** self.t)
            vh = self.v[k] / (1 - self.b2 ** self.t)
            theta[k] += self.lr * mh / (np.sqrt(vh) + self.eps)
