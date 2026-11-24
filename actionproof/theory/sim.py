"""Stochastic trainers on sampled groups for the held-out testbed TH1 (independent of actionproof/ad2 and lab code).

One binary decision (verify) with logit theta; the downstream rule after an alarm is fixed (repair). Multi-decision
behaviour is covered by the existing workflow testbed (docs/research/TESTBED_RESULTS.md cites it). Each step samples B groups of G rollouts under a group law, computes advantages with the chosen
estimator and applies the policy-gradient step averaged over the B*G rollouts (TRL 'dapo' aggregation with one decision
per rollout).

Estimators:
  grpo      (R - mean_g) / (Bessel std_g + 1e-4)
  nostd     R - mean_g                                 (Dr. GRPO's advantage)
  rloo      R - mean_{g,-i}
  batchstd  (R - mean_g) / (std over the whole step's batch + 1e-4)
  reinforce R - b,  b an EMA (0.9) of past steps' mean return  (REINFORCE with a running baseline)
  ppo       (R - b) / batch std, b as reinforce; 4 epochs of clipped surrogate (eps 0.2) on the step's batch
  dapo      grpo with dynamic sampling (zero-variance groups dropped and resampled, up to 4B draws), clip-higher 0.28
            (inactive in its single on-policy epoch)
  grpo_kl   grpo plus beta * KL(pi || pi_ref) penalty, pi_ref = initial policy
  awrb      grpo on E[R | own observation] under the true prior
  gdpo      per-component group normalisation of (utility, cost), summed
Optimisers: plain SGD (comparable to the exact mean-field map `exact.sgd_map`) or Adam.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np

from actionproof.theory.problems import Problem

EPS = 1e-4


def _sig(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


class Sampler:
    def __init__(self, pb: Problem, rb: bool = False):
        self.pb = pb
        self.w = np.array(pb.w) / np.sum(pb.w)
        self.tab = []  # tab[z][r] = (probs, R, util, cost, rbR)
        rbmap = {}
        if rb:
            num, den = {}, {}
            for z, wz in enumerate(pb.w):
                for r in (0, 1):
                    for p, R, o, _ in pb.out[z][r]:
                        num[(r, o)] = num.get((r, o), 0.0) + wz * p * R
                        den[(r, o)] = den.get((r, o), 0.0) + wz * p
            rbmap = {k: num[k] / den[k] for k in num}
        for z in range(pb.K):
            row = []
            for r in (0, 1):
                outs = pb.out[z][r]
                pr = np.array([o[0] for o in outs])
                row.append((pr / pr.sum(), np.array([o[1] for o in outs]), np.array([o[3][0] for o in outs]),
                            np.array([o[3][1] for o in outs]),
                            np.array([rbmap.get((r, o[2]), o[1]) for o in outs])))
            self.tab.append(row)

    def states(self, rng: np.random.Generator, law: str, B: int, G: int) -> np.ndarray:
        K = len(self.w)
        if law == "shared":
            return np.repeat(rng.choice(K, size=(B, 1), p=self.w), G, axis=1)
        if law == "iid":
            return rng.choice(K, size=(B, G), p=self.w)
        if law.startswith("mix:"):
            rho = float(law[4:])
            sh = rng.random(B) < rho
            Z = rng.choice(K, size=(B, G), p=self.w)
            Z[sh] = np.repeat(Z[sh, :1], G, axis=1)
            return Z
        if law.startswith("dirmult:"):
            rho = float(law[8:])
            a0 = (1 - rho) / rho
            al = np.maximum(a0 * self.w, 1e-12)
            Z = np.empty((B, G), dtype=int)
            for b in range(B):
                th = rng.dirichlet(al)
                Z[b] = rng.choice(K, size=G, p=th / th.sum())
            return Z
        raise ValueError(law)

    def outcomes(self, rng: np.random.Generator, Z: np.ndarray, Rd: np.ndarray):
        R = np.empty(Z.shape)
        U = np.empty(Z.shape)
        Cc = np.empty(Z.shape)
        RB = np.empty(Z.shape)
        for z in range(len(self.w)):
            for r in (0, 1):
                m = (Z == z) & (Rd == r)
                n = int(m.sum())
                if n == 0:
                    continue
                pr, Rv, Uv, Cv, RBv = self.tab[z][r]
                idx = rng.choice(len(pr), size=n, p=pr)
                R[m], U[m], Cc[m], RB[m] = Rv[idx], Uv[idx], Cv[idx], RBv[idx]
        return R, U, Cc, RB


def _gnorm(X: np.ndarray, eps: float = EPS) -> np.ndarray:
    mu = X.mean(axis=1, keepdims=True)
    sd = X.std(axis=1, ddof=1, keepdims=True)
    return (X - mu) / (sd + eps)


def train(pb: Problem, G: int, law: str, est: str, steps: int = 300, B: int = 16, lr: float = 0.5, pi0: float = 0.3,
          seed: int = 0, opt: str = "sgd", kl: float = 0.02, log_every: int = 10) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    smp = Sampler(pb, rb=(est == "awrb"))
    th = math.log(pi0 / (1 - pi0))
    th_ref = th
    base = None
    m_adam = v_adam = 0.0
    curve, grads = [], []
    n_rollouts = 0
    for t in range(steps):
        pi = _sig(th)
        if est == "dapo":
            Zs, Rds, Rs = [], [], []
            tries = 0
            while len(Zs) < B and tries < 4 * B:
                Z1 = smp.states(rng, law, 1, G)
                Rd1 = (rng.random((1, G)) < pi).astype(int)
                R1, _, _, _ = smp.outcomes(rng, Z1, Rd1)
                tries += 1
                if R1.std() > 1e-12:
                    Zs.append(Z1)
                    Rds.append(Rd1)
                    Rs.append(R1)
            n_rollouts += tries * G
            if not Zs:
                curve.append(pi) if t % log_every == 0 else None
                continue
            Rd = np.vstack(Rds)
            A = _gnorm(np.vstack(Rs))
        else:
            Z = smp.states(rng, law, B, G)
            Rd = (rng.random((B, G)) < pi).astype(int)
            R, Ut, Cc, RB = smp.outcomes(rng, Z, Rd)
            n_rollouts += B * G
            if est in ("grpo", "grpo_kl"):
                A = _gnorm(R)
            elif est == "awrb":
                A = _gnorm(RB)
            elif est == "nostd":
                A = R - R.mean(axis=1, keepdims=True)
            elif est == "rloo":
                A = (R - R.mean(axis=1, keepdims=True)) * G / (G - 1)
            elif est == "batchstd":
                A = (R - R.mean(axis=1, keepdims=True)) / (R.std(ddof=1) + EPS)
            elif est == "gdpo":
                A = _gnorm(Ut) + _gnorm(Cc)
            elif est in ("reinforce", "ppo"):
                b = R.mean() if base is None else base
                A = R - b
                base = R.mean() if base is None else 0.9 * base + 0.1 * R.mean()
                if est == "ppo":
                    A = A / (A.std() + 1e-8)
            else:
                raise ValueError(est)
        score = Rd - pi
        g = float((A * score).mean())
        grads.append(g)
        if est == "ppo":
            # 4 epochs of the clipped surrogate on this batch, one parameter; gradient of
            # mean_i min(ratio_i A_i, clip(ratio_i) A_i) wrt theta, ratio = pi_theta(r) / pi_old(r)
            pold = np.where(Rd == 1, pi, 1 - pi)
            for _ in range(4):
                pn = _sig(th)
                pr_new = np.where(Rd == 1, pn, 1 - pn)
                ratio = pr_new / pold
                unclipped = ratio * A
                clipped = np.clip(ratio, 0.8, 1.2) * A
                active = unclipped <= clipped  # min picks unclipped -> gradient flows
                dlogp = Rd - pn
                gg = float((np.where(active, ratio * A * dlogp, 0.0)).mean())
                th = _step(th, gg, lr / 4, opt, t)
            if t % log_every == 0:
                curve.append(_sig(th))
            continue
        if est == "grpo_kl":
            g -= kl * pi * (1 - pi) * (th - th_ref)
        if opt == "adam":
            m_adam = 0.9 * m_adam + 0.1 * g
            v_adam = 0.999 * v_adam + 0.001 * g * g
            mh = m_adam / (1 - 0.9 ** (t + 1))
            vh = v_adam / (1 - 0.999 ** (t + 1))
            th += lr * 0.1 * mh / (math.sqrt(vh) + 1e-8)
        else:
            th += lr * g
        th = max(-30.0, min(30.0, th))
        if t % log_every == 0:
            curve.append(_sig(th))
    pi = _sig(th)
    J0, J1 = pb.J(0.0), pb.J(1.0)
    p_fail = sum(wz * ((1 - pi) * _pfail(pb, z, 0) + pi * _pfail(pb, z, 1)) for z, wz in enumerate(pb.w))
    p_fault = sum(wz for z, wz in enumerate(pb.w) if pb.names[z] != "clean")
    rec = (sum(wz * (1 - ((1 - pi) * _pfail(pb, z, 0) + pi * _pfail(pb, z, 1))) for z, wz in enumerate(pb.w)
               if pb.names[z] != "clean") / p_fault) if p_fault > 0 else None
    ga = np.array(grads) if grads else np.zeros(1)
    return {"pi_T": pi, "curve": curve, "J": pb.J(pi), "regret": max(J0, J1) - pb.J(pi),
            "bayes_action": int(J1 > J0), "agrees_bayes": int((pi > 0.5) == (J1 > J0)),
            "false_completion": p_fail, "recovery_given_fault": rec, "rollouts": n_rollouts,
            "grad_mean": float(ga.mean()), "grad_sd": float(ga.std()), "grad_abs_mean": float(np.abs(ga).mean())}


def _step(th: float, g: float, lr: float, opt: str, t: int) -> float:
    return max(-30.0, min(30.0, th + lr * g))


def _pfail(pb: Problem, z: int, r: int) -> float:
    """Probability the task ends without success (the agent always reports completion: false completion)."""
    V = pb.meta.get("V", 1.0)
    return sum(p for p, _, _, comp in pb.out[z][r] if comp[0] < V - 1e-9)
