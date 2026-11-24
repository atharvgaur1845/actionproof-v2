"""Exact expected group-relative updates on one binary decision logit, for any exchangeable group law.

Notation (docs/theory/PROOF_AND_ASSUMPTIONS.md):
  pi = P(r = 1) (verify), score of the logit = r - pi; G rollouts per group; hidden states z with prior w.
  A *group law* is a distribution Q over state compositions m (m_z = number of rollouts of the group in state z),
  exchangeable and with per-rollout marginal w.

Key identity (Proposition 3.1). For any centred estimator (sum_i A_i = 0 in every group) the per-group update
sum_i A_i (r_i - pi) equals sum_{i: r_i = 1} A_i, which does not depend on pi. Hence
    U(pi) = sum_{k=0}^{G} C(G, k) pi^k (1 - pi)^(G - k) b_k,
    b_k  = E[ sum_{readers} A_i | exactly k readers ],   b_0 = b_G = 0,
a Bernstein polynomial whose coefficients carry everything the group law, the stakes and the estimator do.
U'(0) = G b_1 and U'(1) = -G b_{G-1} give the stability of the never- and always-verify corners.

Group laws:
  shared         all G rollouts in one state z ~ w                       (ActionProof as trained)
  iid            each rollout's state drawn independently from w
  mix:rho        with probability rho shared, otherwise iid              (pairwise ICC of state indicators = rho)
  dirmult:rho    Dirichlet-multinomial with concentration (1 - rho)/rho  (pairwise ICC = rho as well)
Estimators: grpo (Bessel std + eps, TRL), grpo_pop, nostd (Dr. GRPO's centring), rloo, gdpo (per-component
normalisation of (utility, cost), summed), awrb (return replaced by E[R | own observation] under the prior, then
grpo), awrbn (the same, centred only).
"""
from __future__ import annotations

import math
from functools import lru_cache
from typing import Callable

import numpy as np

from actionproof.theory.problems import Problem

EPS = 1e-4


# ----------------------------------------------------------------------------------------------- group laws
@lru_cache(maxsize=None)
def compositions(n: int, G: int) -> np.ndarray:
    """All vectors of n non-negative ints summing to G (rows)."""
    if n == 1:
        return np.array([[G]], dtype=np.int16)
    out = []
    for first in range(G, -1, -1):
        rest = compositions(n - 1, G - first)
        out.append(np.hstack([np.full((rest.shape[0], 1), first, dtype=np.int16), rest]))
    return np.vstack(out)


def _log_multinom(m: np.ndarray) -> np.ndarray:
    G = m.sum(axis=1)
    return np.array([math.lgamma(g + 1) for g in G]) - np.vectorize(math.lgamma)(m + 1.0).sum(axis=1)


def law_probs(law: str, w: list[float], G: int) -> tuple[np.ndarray, np.ndarray]:
    """(state compositions M, probabilities Q(M)) for the named group law."""
    K = len(w)
    M = compositions(K, G).astype(np.int64)
    wv = np.array(w, dtype=float)
    with np.errstate(divide="ignore"):
        lw = np.log(wv)

    def iid() -> np.ndarray:
        lp = _log_multinom(M) + np.where(M > 0, M * lw[None, :], 0.0).sum(axis=1)
        return np.exp(lp)

    def shared() -> np.ndarray:
        q = np.zeros(len(M))
        for z in range(K):
            idx = np.where(M[:, z] == G)[0]
            q[idx] = wv[z]
        return q

    if law == "shared":
        q = shared()
    elif law == "iid":
        q = iid()
    elif law.startswith("mix:"):
        rho = float(law[4:])
        q = rho * shared() + (1 - rho) * iid()
    elif law.startswith("dirmult:"):
        rho = float(law[8:])
        if rho >= 1:
            q = shared()
        elif rho <= 0:
            q = iid()
        else:
            a0 = (1 - rho) / rho
            al = a0 * wv
            lg = np.vectorize(math.lgamma)
            lp = _log_multinom(M) + math.lgamma(a0) - math.lgamma(G + a0)
            for z in range(K):
                if wv[z] <= 0:
                    lp = np.where(M[:, z] > 0, -np.inf, lp)
                    continue
                lp = lp + lg(M[:, z] + al[z]) - math.lgamma(al[z])
            q = np.exp(lp)
    else:
        raise ValueError(law)
    q = np.where(np.isfinite(q), q, 0.0)
    return M, q / q.sum()


def pairwise_icc(law: str, w: list[float], G: int) -> list[float]:
    """Pairwise correlation of the indicators 1[z_i = z], 1[z_j = z], per state, computed from Q (a check)."""
    M, q = law_probs(law, w, G)
    out = []
    for z, wz in enumerate(w):
        if wz <= 0 or wz >= 1:
            out.append(float("nan"))
            continue
        pair = (q * M[:, z] * (M[:, z] - 1)).sum() / (G * (G - 1))  # P(z_i = z_j = z)
        out.append(float((pair - wz * wz) / (wz * (1 - wz))))
    return out


# ----------------------------------------------------------------------------------------------- categories
def categories(pb: Problem, est: str) -> dict:
    """Per-rollout categories (z, r, outcome) with conditional probability q = P(outcome | z, r) and the return used by
    the estimator."""
    zs, rs, qs, Rs, comps = [], [], [], [], []
    rb: dict[tuple[int, str], float] = {}
    if est in ("awrb", "awrbn"):
        num: dict[tuple[int, str], float] = {}
        den: dict[tuple[int, str], float] = {}
        for z, wz in enumerate(pb.w):
            for r in (0, 1):
                for p, R, o, _ in pb.out[z][r]:
                    num[(r, o)] = num.get((r, o), 0.0) + wz * p * R
                    den[(r, o)] = den.get((r, o), 0.0) + wz * p
        rb = {k: num[k] / den[k] for k in num if den[k] > 0}
    for z in range(pb.K):
        for r in (0, 1):
            for p, R, o, comp in pb.out[z][r]:
                zs.append(z)
                rs.append(r)
                qs.append(p)
                Rs.append(rb[(r, o)] if rb else R)
                comps.append(comp)
    return {"z": np.array(zs), "r": np.array(rs), "q": np.array(qs), "R": np.array(Rs, dtype=float),
            "comp": np.array(comps, dtype=float)}


def _advantages(Rv: np.ndarray, N: np.ndarray, est: str, eps: float, comp: np.ndarray | None = None) -> np.ndarray:
    """Advantage of each category in each composition (rows: compositions, cols: categories)."""
    G = N.sum(axis=1, keepdims=True).astype(float)

    def centred(vals: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        mu = (N * vals[None, :]).sum(axis=1, keepdims=True) / G
        dev = vals[None, :] - mu
        var = (N * dev ** 2).sum(axis=1, keepdims=True)
        return dev, var

    def scaled(dev: np.ndarray, sd: np.ndarray) -> np.ndarray:
        den = sd + eps
        return np.divide(dev, den, out=np.zeros_like(dev), where=den > 1e-15)

    if est in ("grpo", "awrb"):
        dev, var = centred(Rv)
        return scaled(dev, np.sqrt(var / (G - 1)))
    if est == "grpo_pop":
        dev, var = centred(Rv)
        return scaled(dev, np.sqrt(var / G))
    if est in ("nostd", "awrbn"):
        dev, _ = centred(Rv)
        return dev
    if est == "rloo":
        dev, _ = centred(Rv)
        return dev * G / (G - 1)
    if est == "gdpo":
        tot = 0.0
        for j in range(comp.shape[1]):
            dev, var = centred(comp[:, j])
            tot = tot + scaled(dev, np.sqrt(var / (G - 1)))
        return tot
    raise ValueError(est)


@lru_cache(maxsize=256)
def _cat_compositions(C: int, G: int) -> np.ndarray:
    return compositions(C, G).astype(np.int64)


def bernstein(pb: Problem, G: int, law: str, est: str = "grpo", eps: float = EPS) -> np.ndarray:
    """b_0..b_G: expected group update given exactly k readers (b_0 = b_G = 0)."""
    cat = categories(pb, est)
    C = len(cat["q"])
    N = _cat_compositions(C, G)
    M_law, Q_law = law_probs(law, pb.w, G)
    qindex = {tuple(m): i for i, m in enumerate(M_law)}
    # state composition of every category composition
    Z = np.zeros((C, pb.K), dtype=np.int64)
    Z[np.arange(C), cat["z"]] = 1
    Mn = N @ Z
    qn = np.array([Q_law[qindex[tuple(m)]] for m in Mn])
    keep = qn > 0
    N, Mn, qn = N[keep], Mn[keep], qn[keep]
    lg = np.vectorize(math.lgamma)
    with np.errstate(divide="ignore"):
        lq = np.log(cat["q"])
    logw = (lg(Mn + 1.0).sum(axis=1) - lg(N + 1.0).sum(axis=1)
            + np.where(N > 0, N * lq[None, :], 0.0).sum(axis=1))
    W = qn * np.exp(logw)
    A = _advantages(cat["R"], N, est, eps, cat["comp"])
    u = (N * A * cat["r"][None, :]).sum(axis=1)          # sum over readers of A_i
    k = (N * cat["r"][None, :]).sum(axis=1)
    b = np.zeros(G + 1)
    mass = np.zeros(G + 1)
    np.add.at(b, k, W * u)
    np.add.at(mass, k, W)
    b = np.where(mass > 0, b / np.maximum(mass, 1e-300), 0.0)
    return b


def U_from_b(b: np.ndarray, pi: float | np.ndarray) -> np.ndarray:
    G = len(b) - 1
    arr = np.asarray(pi, dtype=float)
    x = arr.reshape(-1, 1)
    ks = np.arange(G + 1)
    binom = np.array([math.comb(G, int(k)) for k in ks], dtype=float)
    out = (binom[None, :] * x ** ks * (1 - x) ** (G - ks) * b[None, :]).sum(axis=-1)
    return out.reshape(arr.shape)


def expected_update(pb: Problem, pi: float, G: int, law: str, est: str = "grpo", eps: float = EPS) -> float:
    return float(U_from_b(bernstein(pb, G, law, est, eps), pi))


def corner_derivatives(b: np.ndarray) -> dict:
    G = len(b) - 1
    return {"dU_at_0": G * b[1], "dU_at_1": -G * b[G - 1]}


def sign_changes(b: np.ndarray, tol: float = 1e-12) -> int:
    s = [np.sign(x) for x in b[1:-1] if abs(x) > tol]
    return sum(1 for a, c in zip(s, s[1:]) if a != c)


def endpoint(b: np.ndarray, pi0: float, tol: float = 1e-10) -> float:
    """Limit of the mean-field flow d theta/dt = U(pi(theta)) from pi0 (1-D: move with sign U to the first zero)."""
    f = lambda x: float(U_from_b(b, x))
    u0 = f(pi0)
    if abs(u0) < tol:
        return pi0
    grid = np.linspace(pi0, 1.0 if u0 > 0 else 0.0, 4001)[1:]
    vals = U_from_b(b, grid)
    bad = np.where(np.sign(vals) != np.sign(u0))[0]
    bad = bad[(grid[bad] > 1e-9) & (grid[bad] < 1 - 1e-9)]
    if len(bad) == 0:
        return 1.0 if u0 > 0 else 0.0
    j = bad[0]
    lo, hi = (grid[j - 1] if j > 0 else pi0), grid[j]
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if np.sign(f(mid)) == np.sign(u0):
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def sgd_map(b: np.ndarray, pi0: float, steps: int, lr: float, G: int) -> np.ndarray:
    """Deterministic mean-field version of the trainer in sim.py with plain SGD: theta += lr * U(pi) / G per step
    (the sampled gradient is averaged over the B*G rollouts of a step, so its mean is U / G for any B).
    Returns pi_t for t = 0..steps."""
    th = math.log(pi0 / (1 - pi0))
    out = [pi0]
    for _ in range(steps):
        pi = 1 / (1 + math.exp(-th))
        th += lr * float(U_from_b(b, pi)) / G
        th = max(-30.0, min(30.0, th))
        out.append(1 / (1 + math.exp(-th)))
    return np.array(out)


def summary(pb: Problem, G: int, law: str, est: str, pi0: float = 0.3) -> dict:
    b = bernstein(pb, G, law, est)
    pe = endpoint(b, pi0)
    J0, J1 = pb.J(0.0), pb.J(1.0)
    return {"b": b.tolist(), **corner_derivatives(b), "sign_changes": sign_changes(b), "endpoint": pe,
            "U_pi0": float(U_from_b(b, pi0)), "regret": max(J0, J1) - pb.J(pe), "delta_q": pb.delta_q(),
            "vote": pb.vote()}
