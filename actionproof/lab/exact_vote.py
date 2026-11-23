"""Exact (enumeration, no Monte Carlo) expected group-relative update on a verification logit.

Model (docs/research_lab/GRPO_MECHANISM_FORMAL.md §2). One aliased verify-or-skip decision on a write that is silently
faulted with probability p (the latent fault z in {0 clean, 1 faulted}). Per rollout i:
    r_i ~ Bern(rho)                     read (verify) or not; the policy's only stochastic choice here
    u_i ~ Bern(s)                       a repair after a detected fault succeeds (rollout randomness)
    e_i ~ Bern(q0)                      exogenous task failure, independent of everything (rollout randomness)
    succ_i = (1 - e_i) * (1 if z = 0 else r_i * u_i)
    R_i = V succ_i - c r_i - Lf (1 - succ_i)
Score of the read logit: d/dlogit log pi(r_i) = r_i - rho.

Group conditions: 'A' all G rollouts share one z (ActionProof as trained); 'B' z i.i.d. per rollout.
Estimators: 'grpo' (R - mean) / (Bessel std + eps); 'grpo_pop' (population std + eps); 'nostd' R - mean;
'rloo' leave-one-out.  Expected update  E[ sum_i A_i (r_i - rho) ]  is computed exactly by summing over all
multinomial compositions of the G rollouts over the outcome categories.
"""
from __future__ import annotations

import itertools
import math
from functools import lru_cache

import numpy as np


def categories(p: float, c: float, s: float, rho: float, V: float = 1.0, Lf: float = 0.0, q0: float = 0.0,
               world: int | None = None) -> list[tuple[float, float, float]]:
    """Outcome categories (probability, return, score) of one rollout; world=None mixes z ~ Bern(p)."""
    out = []
    worlds = [(0, 1 - p), (1, p)] if world is None else [(world, 1.0)]
    for z, pz in worlds:
        if pz <= 0:
            continue
        for r, pr in ((0, 1 - rho), (1, rho)):
            for u, pu in ((0, 1 - s), (1, s)):
                for e, pe in ((0, 1 - q0), (1, q0)):
                    w = pz * pr * pu * pe
                    if w <= 0:
                        continue
                    succ = (1 - e) * (1 if z == 0 else r * u)
                    R = V * succ - c * r - Lf * (1 - succ)
                    out.append((w, R, r - rho))
    # merge identical (R, score) categories
    merged: dict[tuple[float, float], float] = {}
    for w, R, sc in out:
        k = (round(R, 12), round(sc, 12))
        merged[k] = merged.get(k, 0.0) + w
    return [(w, R, sc) for (R, sc), w in merged.items()]


@lru_cache(maxsize=None)
def _compositions(n_cat: int, G: int) -> tuple[tuple[int, ...], ...]:
    out = []
    for bars in itertools.combinations(range(G + n_cat - 1), n_cat - 1):
        prev, comp = -1, []
        for b in bars:
            comp.append(b - prev - 1)
            prev = b
        comp.append(G + n_cat - 1 - prev - 1)
        out.append(tuple(comp))
    return tuple(out)


def _group_update(R: np.ndarray, S: np.ndarray, est: str, eps: float) -> float:
    G = len(R)
    mu = R.mean()
    if est == "grpo":
        A = (R - mu) / (R.std(ddof=1) + eps)
    elif est == "grpo_pop":
        A = (R - mu) / (R.std() + eps)
    elif est == "nostd":
        A = R - mu
    elif est == "rloo":
        A = R - (R.sum() - R) / (G - 1)
    else:
        raise ValueError(est)
    return float((A * S).sum())


def expected_update_cats(cats: list[tuple[float, float, float]], G: int, est: str, eps: float = 1e-4) -> float:
    w = np.array([x[0] for x in cats])
    Rv = np.array([x[1] for x in cats])
    Sv = np.array([x[2] for x in cats])
    tot = 0.0
    logG = math.lgamma(G + 1)
    for comp in _compositions(len(cats), G):
        lp = logG
        for k, wk in zip(comp, w):
            if k:
                lp += k * math.log(wk) - math.lgamma(k + 1)
        idx = np.repeat(np.arange(len(cats)), comp)
        tot += math.exp(lp) * _group_update(Rv[idx], Sv[idx], est, eps)
    return tot


def expected_update(p: float, c: float, s: float, rho: float, G: int = 8, est: str = "grpo", cond: str = "A",
                    V: float = 1.0, Lf: float = 0.0, q0: float = 0.0, eps: float = 1e-4) -> float:
    if cond == "A":
        tot = 0.0
        for z, pz in ((0, 1 - p), (1, p)):
            if pz > 0:
                tot += pz * expected_update_cats(categories(p, c, s, rho, V, Lf, q0, world=z), G, est, eps)
        return tot
    return expected_update_cats(categories(p, c, s, rho, V, Lf, q0), G, est, eps)


def voi(p: float, c: float, s: float, V: float = 1.0, Lf: float = 0.0, q0: float = 0.0) -> float:
    """Exact Q(verify) - Q(skip) of the single decision (repair always attempted after a detected fault)."""
    return p * s * (1 - q0) * (V + Lf) - c


def threshold(c: float, s: float, rho: float, G: int = 8, est: str = "grpo", cond: str = "A", V: float = 1.0,
              Lf: float = 0.0, q0: float = 0.0, eps: float = 1e-4, tol: float = 1e-4) -> float | None:
    """Smallest fault rate p in (0, 1) at which the expected update on the read logit is positive (bisection on the
    sign change; under condition A the update is linear in p, under B a polynomial). None if never positive."""
    f = lambda p: expected_update(p, c, s, rho, G, est, cond, V, Lf, q0, eps)
    lo, hi = 1e-6, 1 - 1e-6
    flo, fhi = f(lo), f(hi)
    if flo > 0:
        return 0.0
    if fhi <= 0:
        return None
    if cond == "A":  # linear in p: exact root
        return float(lo + (hi - lo) * (-flo) / (fhi - flo))
    while hi - lo > tol:
        m = 0.5 * (lo + hi)
        if f(m) > 0:
            hi = m
        else:
            lo = m
    return 0.5 * (lo + hi)


def m_G(rho: float, G: int) -> float:
    """E[ sd_Bessel(r_1..r_G) ; 0 < k < G ],  k ~ Bin(G, rho): the size of one world's vote (Theorem 1)."""
    tot = 0.0
    for k in range(1, G):
        tot += math.comb(G, k) * rho ** k * (1 - rho) ** (G - k) * math.sqrt(k * (G - k) / (G * (G - 1)))
    return tot
