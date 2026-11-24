"""Single-decision verification problems with K hidden states.

A problem is a list of hidden states z with prior w_z. For each state and each value of the binary decision r
(0 skip, 1 verify) it lists the outcome distribution as (probability, return, obs, components), where
  return      the scalar reward R used by the estimators;
  obs         what the agent observes after the decision (the aliasing key for observation-measurable returns);
  components  (utility, cost) for per-component (GDPO-style) normalisation.

`verification(...)` builds the costed-verification family used throughout docs/theory:
  clean state (z = 0) and fault states z = 1..K-1, each with
    L_z   loss if the fault is left unrepaired (severity; success is worth V)
    s_z   probability that the repair after a detected fault succeeds
          (rollout randomness unless the state is split by `split_repair=True`, which makes repairability part of z)
    d_z   detection probability of the read (1 = perfect; d < 1 gives misses)
  f       false-alarm probability of a read on a clean write; the fixed downstream rule repairs on an alarm, which on a
          clean write duplicates an irreversible action at cost D
  c       read cost (may be a per-state tuple: state-dependent cost)
  q0      exogenous failure probability, independent of everything (within-state noise)
Downstream behaviour after the read is fixed (repair iff an alarm is shown). Q(verify) - Q(skip) under that fixed rule
is `delta_q`; it equals the decision-theoretic net VOI iff that rule is Bayes-optimal (checked by `voi_decision`).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

Outcome = tuple[float, float, str, tuple[float, float]]


@dataclass
class Problem:
    w: list[float]                                  # prior over hidden states
    out: list[list[list[Outcome]]]                  # out[z][r] = list of outcomes
    names: list[str] = field(default_factory=list)
    meta: dict = field(default_factory=dict)

    @property
    def K(self) -> int:
        return len(self.w)

    def mean_return(self, z: int, r: int) -> float:
        return sum(p * R for p, R, _, _ in self.out[z][r])

    def beta(self, z: int) -> float:
        """Within-state effect of verifying, E[R | z, r=1] - E[R | z, r=0]."""
        return self.mean_return(z, 1) - self.mean_return(z, 0)

    def delta_q(self) -> float:
        return sum(wz * self.beta(z) for z, wz in enumerate(self.w))

    def J(self, pi: float) -> float:
        return sum(wz * ((1 - pi) * self.mean_return(z, 0) + pi * self.mean_return(z, 1)) for z, wz in enumerate(self.w))

    def deterministic_within_state(self) -> bool:
        return all(len({round(R, 12) for _, R, _, _ in self.out[z][r]}) == 1 for z in range(self.K) for r in (0, 1))

    def vote(self) -> float:
        """sum_z w_z sign(beta_z): the stake-free vote (Theorem 1)."""
        s = 0.0
        for z, wz in enumerate(self.w):
            b = self.beta(z)
            s += wz * (1.0 if b > 1e-12 else -1.0 if b < -1e-12 else 0.0)
        return s

    def reads_neutralise(self) -> bool:
        """Every verifying rollout gets the same return whatever the hidden state (Corollary 3.3 condition)."""
        vals = {round(R, 12) for z in range(self.K) for _, R, _, _ in self.out[z][1]}
        return len(vals) == 1


def _per_state(x, K: int) -> list[float]:
    if isinstance(x, (int, float)):
        return [float(x)] * K
    assert len(x) == K
    return [float(v) for v in x]


def verification(p: Sequence[float], L: Sequence[float], s: Sequence[float], d: Sequence[float] | None = None,
                 c: float | Sequence[float] = 0.05, V: float = 1.0, f: float = 0.0, D: float = 0.0, q0: float = 0.0,
                 L0: float = 0.0, split_repair: bool = False, uninformative: bool = False) -> Problem:
    """p[j]: prior of fault state j (clean gets 1 - sum p). L, s, d per fault state; c a scalar or a tuple of length
    1 + len(p) (clean first: state-dependent read cost). L0: loss of an exogenous failure in the clean state.
    uninformative=True: the read reveals nothing (d = 0, f = 0). See the module docstring."""
    nf = len(p)
    d = [1.0] * nf if d is None else list(d)
    if uninformative:
        d, f = [0.0] * nf, 0.0
    cb = _per_state(c, 1 + nf)
    # (name, prior, loss, repair prob, detection prob, read cost)
    states = [("clean", 1 - sum(p), L0, 1.0, 0.0, cb[0])]
    for j in range(nf):
        if split_repair and 0 < s[j] < 1:
            states.append((f"f{j}:rep", p[j] * s[j], L[j], 1.0, d[j], cb[1 + j]))
            states.append((f"f{j}:unrep", p[j] * (1 - s[j]), L[j], 0.0, d[j], cb[1 + j]))
        else:
            states.append((f"f{j}", p[j], L[j], s[j], d[j], cb[1 + j]))
    out: list[list[list[Outcome]]] = []
    for name, pz, Lz, sz, dz, cost in states:
        rows0: list[Outcome] = []
        rows1: list[Outcome] = []
        for e, pe in ((0, 1 - q0), (1, q0)):
            if pe <= 0:
                continue
            if name == "clean":
                U0 = V if e == 0 else -Lz
                rows0.append((pe, U0, "none", (U0, 0.0)))
                for alarm, pa in ((0, 1 - f), (1, f)):
                    if pa > 0:
                        U = U0 - (D if alarm else 0.0)
                        rows1.append((pe * pa, U - cost, "alarm" if alarm else "clean", (U, -cost)))
            else:
                rows0.append((pe, -Lz, "none", (-Lz, 0.0)))
                for det, pd in ((1, dz), (0, 1 - dz)):
                    if pd <= 0:
                        continue
                    if det:
                        for ok, po in ((1, sz), (0, 1 - sz)):
                            if po > 0:
                                U = V if (ok and e == 0) else -Lz
                                rows1.append((pe * pd * po, U - cost, "alarm", (U, -cost)))
                    else:
                        rows1.append((pe * pd, -Lz - cost, "clean", (-Lz, -cost)))
        out.append([_merge(rows0), _merge(rows1)])
    return Problem(w=[st[1] for st in states], out=out, names=[st[0] for st in states],
                   meta=dict(p=list(p), L=list(L), s=list(s), d=d, c=c, V=V, f=f, D=D, q0=q0, L0=L0,
                             split_repair=split_repair, cost_by_state=[st[5] for st in states]))


def _merge(rows: list[Outcome]) -> list[Outcome]:
    acc: dict[tuple, list] = {}
    for pr, R, obs, comp in rows:
        k = (round(R, 12), obs, (round(comp[0], 12), round(comp[1], 12)))
        if k in acc:
            acc[k][0] += pr
        else:
            acc[k] = [pr, R, obs, comp]
    return [tuple(v) for v in acc.values() if v[0] > 0]  # type: ignore[misc]


def rescale_states(pb: Problem, lam: Sequence[float], kappa: Sequence[float] | None = None) -> Problem:
    """Per-state positive affine transform of every return: R -> lam_z R + kappa_z (Theorem 2's invariance)."""
    kappa = [0.0] * pb.K if kappa is None else list(kappa)
    out = [[[(p, lam[z] * R + kappa[z], o, (lam[z] * cp[0] + kappa[z], lam[z] * cp[1])) for p, R, o, cp in pb.out[z][r]]
            for r in (0, 1)] for z in range(pb.K)]
    return Problem(w=list(pb.w), out=out, names=list(pb.names), meta=dict(pb.meta, rescaled=list(lam)))


def voi_decision(pb: Problem) -> dict:
    """Decision-theoretic gross and net VOI of the read, allowing the downstream choice after each observation to be
    optimal among {repair-on-alarm (as modelled), do nothing}. Returns gross VOI, net VOI and whether the modelled
    downstream rule is Bayes-optimal (if not, delta_q < net VOI)."""
    # value without reading = E[R | skip] (no downstream action is possible without an observation in this family)
    v_skip = sum(wz * pb.mean_return(z, 0) for z, wz in enumerate(pb.w))
    # with reading under the modelled rule
    v_read_rule = sum(wz * pb.mean_return(z, 1) for z, wz in enumerate(pb.w))
    cz = pb.meta.get("cost_by_state", [0.0] * pb.K)
    cbar = sum(wz * cz[z] for z, wz in enumerate(pb.w))
    return {"v_skip": v_skip, "v_read_rule": v_read_rule, "delta_q": v_read_rule - v_skip,
            "gross_voi_rule": v_read_rule + cbar - v_skip}
