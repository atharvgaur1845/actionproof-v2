"""Controlled costed-verification testbed (CPU) with an exactly solvable optimum.

A workflow is a sequence of K write operations. Each operation has an observable type with parameters:
  p     probability of a silent fault behind an "ok" acknowledgement
  s     probability that a repair after a detected silent fault succeeds
  f     false-alarm probability of a verification read on a clean write
  q_to  probability of a visible "timeout" acknowledgement
  a     probability that a timed-out write was in fact applied (blind retry then duplicates it)

Hidden fault plan (the "world"): z_j in {0 clean, 1 silent fault, 2 timeout-applied, 3 timeout-not-applied}.
  mode "multi":  independent per operation.
  mode "single": at most one silent fault per workflow (P(silent at j) = p_j); timeouts independent.
Rollout randomness (not part of the world, resampled per rollout as LLM sampling noise is):
  ro_j ~ Bern(s_j) (repair would succeed), fa_j ~ Bern(f_j) (a read of clean write j raises a false alarm).

Decision points (observable information only; the policy never sees z):
  V1  after an ok ack                  {skip, verify}
  R   after a read that showed a fault  {proceed, repair}
  V2  after a repair                    {stop, re-verify}   (shows whether the repair held; enables early stop)
  V2c after a read that showed clean    {stop, re-verify}   (zero information by construction)
  T   after a timeout ack               {proceed, verify, blind retry}
  TR  after a read showing not applied  {proceed, retry}
  TRa after a read showing applied      {proceed, retry}    (retry duplicates)
Observable context feature kf ("known failed"): the agent has seen an unresolved fault indication.

Return of a rollout:  R = V*succ - c*n_verify - Lf*(1-succ) - D*n_dup
  verified task utility V*succ, verification cost c*n_verify, failure loss Lf*(1-succ), duplicate loss D*n_dup.

Everything is computed on a set of rows (worlds x rollout randomness), so the same walker gives
  - an exact expectimax optimum and exact policy values (rows = every combination, weighted by probability);
  - a sampled rollout together with its observation-consistent world set (rows = every fault plan with the rollout's
    own randomness), hence the Rao-Blackwellised return E[R | trajectory] under any assumed prior.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

DTYPES = {"V1": 2, "R": 2, "V2": 2, "V2c": 2, "T": 3, "TR": 2, "TRa": 2}


@dataclass(frozen=True)
class OpType:
    name: str
    p: float
    s: float = 1.0
    f: float = 0.0
    q_to: float = 0.0
    a: float = 0.5
    tid: int = 0  # index into the policy's type one-hot


@dataclass
class Config:
    ops: list[OpType]
    c: float = 0.05
    V: float = 1.0
    Lf: float = 0.0
    D: float = 0.0
    mode: str = "multi"
    tag: str = ""

    @property
    def K(self) -> int:
        return len(self.ops)


def fault_plans(cfg: Config) -> tuple[np.ndarray, np.ndarray]:
    """All fault plans z (n x K) with prior probabilities (n,)."""
    K = cfg.K
    Z, P = [], []
    for z in itertools.product(range(4), repeat=K):
        pr = 1.0
        n_silent = sum(1 for x in z if x == 1)
        if cfg.mode == "single" and n_silent > 1:
            continue
        for j, x in enumerate(z):
            o = cfg.ops[j]
            if x >= 2:
                pr *= o.q_to * (o.a if x == 2 else 1 - o.a)
            else:
                pr *= 1 - o.q_to
        if cfg.mode == "multi":
            for j, x in enumerate(z):
                if x < 2:
                    pr *= cfg.ops[j].p if x == 1 else 1 - cfg.ops[j].p
        else:
            ok_ack = [j for j, x in enumerate(z) if x < 2]
            sil = [j for j in ok_ack if z[j] == 1]
            if sil:
                pr *= cfg.ops[sil[0]].p
            else:
                pr *= max(0.0, 1 - sum(cfg.ops[j].p for j in ok_ack))
        if pr > 0:
            Z.append(z)
            P.append(pr)
    Z, P = np.array(Z, dtype=np.int8).reshape(-1, K), np.array(P)
    return Z, P / P.sum()


@dataclass
class Rows:
    Z: np.ndarray      # n x K fault plans
    RO: np.ndarray     # n x K repair-success bits
    FA: np.ndarray     # n x K false-alarm bits
    w: np.ndarray      # n   probability mass (prior x randomness), possibly unnormalised
    actual: int = -1   # sampled mode: index of the actual row (observations come from it); -1 = exact mode


def exact_rows(cfg: Config, prior: tuple[np.ndarray, np.ndarray] | None = None) -> Rows:
    Z, P = prior or fault_plans(cfg)
    K = cfg.K
    bits = np.array(list(itertools.product((0, 1), repeat=K)), dtype=np.int8).reshape(-1, K)
    s = np.array([o.s for o in cfg.ops])
    f = np.array([o.f for o in cfg.ops])
    pro = np.prod(np.where(bits == 1, s, 1 - s), axis=1)
    pfa = np.prod(np.where(bits == 1, f, 1 - f), axis=1)
    keep_ro, keep_fa = pro > 0, pfa > 0
    br, bf = bits[keep_ro], bits[keep_fa]
    pro, pfa = pro[keep_ro], pfa[keep_fa]
    nz, nr, nf = len(Z), len(br), len(bf)
    iz, ir, i_f = np.meshgrid(np.arange(nz), np.arange(nr), np.arange(nf), indexing="ij")
    iz, ir, i_f = iz.ravel(), ir.ravel(), i_f.ravel()
    return Rows(Z[iz], br[ir], bf[i_f], P[iz] * pro[ir] * pfa[i_f])


def sample_rows(cfg: Config, rng: np.random.Generator, z_actual: np.ndarray,
                prior: tuple[np.ndarray, np.ndarray]) -> Rows:
    """Every fault plan of `prior` with this rollout's own randomness; observations come from z_actual."""
    Z, P = prior
    K = cfg.K
    ro = (rng.random(K) < np.array([o.s for o in cfg.ops])).astype(np.int8)
    fa = (rng.random(K) < np.array([o.f for o in cfg.ops])).astype(np.int8)
    idx = np.where((Z == z_actual).all(axis=1))[0]
    if len(idx) == 0:  # actual world outside the assumed prior's support: append it with zero assumed mass
        Z = np.vstack([Z, z_actual[None]])
        P = np.append(P, 0.0)
        idx = [len(Z) - 1]
    n = len(Z)
    return Rows(Z, np.tile(ro, (n, 1)), np.tile(fa, (n, 1)), P.copy(), actual=int(idx[0]))


def sample_world(cfg: Config, rng: np.random.Generator, true_prior: tuple[np.ndarray, np.ndarray]) -> np.ndarray:
    Z, P = true_prior
    return Z[rng.choice(len(Z), p=P)]


# --------------------------------------------------------------------------------------------- the walker
@dataclass
class State:
    rows: Rows
    alive: np.ndarray            # bool n: consistent with every observation so far
    correct: np.ndarray          # bool n x K
    dup: np.ndarray              # int n
    nv: int = 0
    kf: bool = False
    j: int = 0
    log: list = field(default_factory=list)  # sampled mode: decision records

    def copy(self) -> "State":
        return State(self.rows, self.alive.copy(), self.correct.copy(), self.dup.copy(), self.nv, self.kf, self.j,
                     list(self.log))


def init_state(cfg: Config, rows: Rows) -> State:
    n = len(rows.w)
    return State(rows, np.ones(n, bool), np.zeros((n, cfg.K), bool), np.zeros(n, np.int32))


def returns(cfg: Config, st: State) -> dict[str, np.ndarray]:
    succ = st.correct.all(axis=1)
    U = cfg.V * succ
    cost = cfg.c * st.nv * np.ones(len(succ))
    fail = cfg.Lf * (~succ)
    dupl = cfg.D * st.dup
    return {"R": U - cost - fail - dupl, "U": U, "cost": cost, "fail": fail, "dupl": dupl, "succ": succ}


Chooser = Callable[[str, dict[str, Any], State], Any]


def _split(st: State, obs: np.ndarray) -> list[tuple[Any, State]]:
    """Branch on an observation (array over rows). Exact mode: every value with mass; sampled: the actual one."""
    if st.rows.actual >= 0:
        v = obs[st.rows.actual]
        s2 = st
        s2.alive = st.alive & (obs == v)
        return [(v, s2)]
    out = []
    for v in np.unique(obs[st.alive]):
        s2 = st.copy()
        s2.alive = st.alive & (obs == v)
        if (st.rows.w * s2.alive).sum() > 0:
            out.append((v, s2))
    return out


def ctx(cfg: Config, st: State, j: int) -> dict[str, Any]:
    o = cfg.ops[j]
    return {"op": o, "kf": st.kf, "c": cfg.c, "j": j, "cfg": cfg}


def walk(cfg: Config, st: State, choose: Chooser, combine: str) -> Any:
    """combine 'leaf': sampled mode, choose() returns one action, walk returns the final State.
    combine 'mixed': exact mode, choose() returns a distribution over actions (policy expectation) or None (take the
    best action: expectimax). walk returns the expected return (probability-weighted sum over rows)."""
    if st.j >= cfg.K:
        if combine == "leaf":
            return st
        r = returns(cfg, st)["R"]
        return float((st.rows.w * st.alive * r).sum())
    j = st.j
    Z = st.rows.Z[:, j]
    branches = _split(st, Z >= 2)  # ack: timeout or ok
    vals = []
    for timeout, s in branches:
        vals.append(_op_timeout(cfg, s, j, choose, combine) if timeout else _op_ok(cfg, s, j, choose, combine))
    if combine == "leaf":
        return vals[0]
    return sum(vals)


def _decide(cfg, st, dtype, j, choose, combine, cont):
    """cont(action, state) -> value/leaf."""
    c = ctx(cfg, st, j)
    n = DTYPES[dtype]
    if combine == "leaf":
        return cont(choose(dtype, c, st), st)
    pr = choose(dtype, c, st)
    if pr is None:
        return max(cont(a, st.copy()) for a in range(n))
    return sum(pr[a] * cont(a, st.copy()) for a in range(n) if pr[a] > 0)


def _next(cfg, st, choose, combine):
    st.j += 1
    return walk(cfg, st, choose, combine)


def _op_ok(cfg, st, j, choose, combine):
    Z = st.rows.Z[:, j]

    def after_v1(a, s):
        if a == 0:  # skip
            s.correct[:, j] = Z == 0
            return _next(cfg, s, choose, combine)
        s.nv += 1
        flag = (Z == 1) | ((Z == 0) & (s.rows.FA[:, j] == 1))
        res = []
        for fault_seen, s2 in _split(s, flag):
            res.append(after_read(bool(fault_seen), s2))
        return res[0] if combine == "leaf" else sum(res)

    def after_read(fault_seen, s):
        if not fault_seen:
            s.correct[:, j] = True  # rows still alive here are clean without false alarm
            return _decide(cfg, s, "V2c", j, choose, combine, lambda a, s3: reverify(a, s3, clean=True))
        return _decide(cfg, s, "R", j, choose, combine, after_r)

    def after_r(a, s):
        if a == 0:  # proceed without repair
            s.correct[:, j] = Z == 0  # a false alarm left a correct write
            s.kf = True
            return _next(cfg, s, choose, combine)
        fixed = np.where(Z == 1, s.rows.RO[:, j] == 1, True)
        s.dup = s.dup + ((Z == 0) & True).astype(np.int32)  # repairing a clean write duplicates it
        s.correct[:, j] = fixed
        return _decide(cfg, s, "V2", j, choose, combine, lambda a2, s3: reverify(a2, s3, clean=False))

    def reverify(a, s, clean):
        if a == 0:
            return _next(cfg, s, choose, combine)
        s.nv += 1
        res = []
        for ok, s2 in _split(s, s.correct[:, j].copy()):
            if not ok:
                s2.kf = True
            res.append(_next(cfg, s2, choose, combine))
        return res[0] if combine == "leaf" else sum(res)

    return _decide(cfg, st, "V1", j, choose, combine, after_v1)


def _op_timeout(cfg, st, j, choose, combine):
    Z = st.rows.Z[:, j]

    def after_t(a, s):
        if a == 0:  # proceed
            s.correct[:, j] = Z == 2
            return _next(cfg, s, choose, combine)
        if a == 2:  # blind retry
            s.dup = s.dup + (Z == 2).astype(np.int32)
            s.correct[:, j] = True
            return _next(cfg, s, choose, combine)
        s.nv += 1
        res = []
        for applied, s2 in _split(s, Z == 2):
            dt = "TRa" if applied else "TR"
            res.append(_decide(cfg, s2, dt, j, choose, combine, lambda a2, s3, ap=bool(applied): after_tr(a2, s3, ap)))
        return res[0] if combine == "leaf" else sum(res)

    def after_tr(a, s, applied):
        if applied:
            if a == 1:
                s.dup = s.dup + 1
            s.correct[:, j] = True
        else:
            s.correct[:, j] = a == 1
            if a == 0:
                s.kf = True
        return _next(cfg, s, choose, combine)

    return _decide(cfg, st, "T", j, choose, combine, after_t)


# --------------------------------------------------------------------------------------------- exact solvers
def oracle_value(cfg: Config, rows: Rows | None = None) -> float:
    rows = rows or exact_rows(cfg)
    return walk(cfg, init_state(cfg, rows), lambda *a: None, "mixed")


def policy_value(cfg: Config, probs: Callable[[str, dict[str, Any]], np.ndarray], rows: Rows | None = None) -> float:
    rows = rows or exact_rows(cfg)
    return walk(cfg, init_state(cfg, rows), lambda d, c, s: probs(d, c), "mixed")


def root_voi(cfg: Config, rows: Rows | None = None) -> list[dict[str, float]]:
    """For each op j: Q(verify) - Q(skip) at its ok-ack decision on the canonical history 'every earlier decision
    skipped (timeouts: verified)', with the optimal continuation afterwards. The expected value of information of
    verifying op j, net of its cost; > 0 means the optimal policy verifies there."""
    rows = rows or exact_rows(cfg)
    out = []
    for j in range(cfg.K):
        q = {}
        for a in (0, 1):
            def ch(dtype, c, s, a=a, j=j):
                if c["j"] < j:
                    return np.array([0, 1.0, 0]) if dtype == "T" else np.eye(DTYPES[dtype])[0]
                if c["j"] == j and dtype == "V1":
                    return np.eye(2)[a]
                return None
            q[a] = walk(cfg, init_state(cfg, rows), ch, "mixed")
        out.append({"op": cfg.ops[j].name, "j": j, "q_skip": q[0], "q_verify": q[1], "voi": q[1] - q[0]})
    return out


def sample_episode(cfg: Config, rng: np.random.Generator, z_actual: np.ndarray, assumed_prior, policy_sample) -> dict:
    """One rollout in world z_actual. policy_sample(dtype, ctx, state, rng) -> (action, record dict).
    Returns per-rollout returns, the Rao-Blackwellised return under `assumed_prior` and the decision log."""
    rows = sample_rows(cfg, rng, z_actual, assumed_prior)
    st = init_state(cfg, rows)

    def choose(dtype, c, s):
        a, rec = policy_sample(dtype, c, s, rng)
        s.log.append(rec)
        return a
    st = walk(cfg, st, choose, "leaf")
    r = returns(cfg, st)
    i = rows.actual
    m = rows.w * st.alive
    den = m.sum()
    rbar = float((m * r["R"]).sum() / den) if den > 0 else float(r["R"][i])
    return {"R": float(r["R"][i]), "U": float(r["U"][i]), "cost": float(r["cost"][i]), "fail": float(r["fail"][i]),
            "dupl": float(r["dupl"][i]), "succ": bool(r["succ"][i]), "rbar": rbar, "rb_defined": bool(den > 0),
            "nv": st.nv, "log": st.log, "z": z_actual, "state": st}
