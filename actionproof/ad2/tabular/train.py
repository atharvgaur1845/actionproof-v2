"""Policy-gradient estimators on the tabular costed-verification testbed.

Every estimator optimises the same objective J = E[R] (R = V*succ - c*reads - Lf*fail - D*dups) from the same sampled
groups (G rollouts per task, `groups` tasks per step), the same policy class, optimiser, learning rate and seed.
The only differences are the return used, the advantage, and (for the decision-level estimators) extra fork
rollouts, which are counted in `episodes_used` so that budgets can be matched.

Sequence-level (A_i multiplies every decision of rollout i; gradient summed over decisions and divided by the
number of decisions in the batch, as TRL's 'dapo' token-level loss does):
  grpo       (R - mean_g) / (std_g + 1e-4)            TRL scale_rewards='group' (Bessel std)
  nostd      R - mean_g                               Dr. GRPO's std removal (scale_rewards='none')
  rloo       R - mean_{g, -i}                         leave-one-out baseline (Kool et al.; Ahmadian et al.)
  batchstd   (R - mean_g) / (std_batch + 1e-4)        TRL scale_rewards='batch' / REINFORCE++ global norm
  gdpo       per-component group normalisation, summed, then batch normalised (TRL normalize_then_sum; GDPO)
  awrb       (Rbar - mean_g) / (std_g + 1e-4)         first-stage AWRB, Rbar = E_assumed[R | trajectory]
  awrbn      Rbar - mean_g                            first-stage AWRB-n
  lagr       nostd on U - fail - dup - lam*c*reads, lam by dual ascent on E[c*reads] <= budget
  lagr_grpo  the same Lagrangian reward under group std (constrained GRPO)
  ema        (R - mean_g) / s_t, s_t = max(0.02, EMA_0.9 of past steps' mean group std)   (first-stage D2)
  ctxema     as ema, but one EMA per observable task context (cfg.tag), updated after the step that uses it:
             a per-prompt-type scale that never depends on the current group's hidden world
Decision-level (each verification decision gets its own advantage; other decisions use the nostd sequence term):
  critic_v   R - Vhat(observable key)                 REINFORCE / actor-critic with a learned state-value baseline
  critic_q   all-action gradient with a learned Qhat(observable key, a)  (mean actor-critic)
  igc        uninformative reads get the exact local value -c (first-stage D3 idea); others sequence nostd
  fork_act   all-action gradient, Q(h,a) by K forked continuations in the *actual* hidden world (VinePPO/TreeRL)
  fork_bel   all-action gradient, Q(h,a) by K forked continuations in worlds resampled from the *posterior*
             b(z | h) (POMCP-style root sampling); common random numbers across the two actions
  fork_bel_std  fork_bel with the per-decision Q-difference divided by its fork-group std (tests whether a
             normaliser estimated from the same forks reintroduces world dependence)
  cvc        counterfactual verification credit: uninformative verification decisions get the exact local value
             (verify = -c, no forks); informative ones get belief forks (fork_bel); others sequence nostd
"""
from __future__ import annotations

import copy
import math
from collections import defaultdict
from dataclasses import replace
from typing import Any, Callable

import numpy as np

from actionproof.ad2.tabular.env import (DTYPES, Config, OpType, Rows, State, _op_ok, _op_timeout, fault_plans,
                                         oracle_value, policy_value, returns, root_voi, sample_episode, sample_world,
                                         walk)
from actionproof.ad2.tabular.policy import Adam, Policy

SEQ = {"grpo", "nostd", "rloo", "batchstd", "gdpo", "awrb", "awrbn", "lagr", "lagr_grpo", "ema", "ctxema"}
DEC = {"critic_v", "critic_q", "igc", "fork_act", "fork_bel", "fork_bel_std", "cvc"}
VERIFY_D = ("V1", "T")


# --------------------------------------------------------------------------------------------- priors
_PRIOR_CACHE: dict = {}


def prior_of(cfg: Config) -> tuple[np.ndarray, np.ndarray]:
    key = (tuple(cfg.ops), cfg.mode)
    if key not in _PRIOR_CACHE:
        _PRIOR_CACHE[key] = fault_plans(cfg)
    return _PRIOR_CACHE[key]


def assumed_cfg(cfg: Config, spec: str, phat: dict[int, float] | None = None) -> Config:
    if spec == "true":
        return cfg
    if spec == "est":
        ops = [replace(o, p=min(0.95, phat.get(o.tid, 0.5))) for o in cfg.ops]
        return replace(cfg, ops=ops)
    if spec.startswith("mis:"):
        k = float(spec[4:])
        ops = [replace(o, p=min(0.95, o.p * k)) for o in cfg.ops]
        return replace(cfg, ops=ops)
    if spec.startswith("flat:"):  # every op type assumed to have fault rate x (wrong for the safe op, too)
        x = float(spec[5:])
        return replace(cfg, ops=[replace(o, p=x) for o in cfg.ops])
    raise ValueError(spec)


# --------------------------------------------------------------------------------------------- forks
def fork_return(cfg: Config, rec: dict, pol: Policy, rng: np.random.Generator, action: int, world: str,
                seed: int, z_actual: np.ndarray) -> float:
    """Continue from the decision state of `rec`, forcing `action`, then following the policy. The hidden world is
    the actual one (world='act') or drawn from the posterior over alive rows (world='bel'). Future randomness is
    drawn from `seed`, shared by the two actions (common random numbers)."""
    st: State = rec["state"]
    r = np.random.default_rng(seed)
    m = st.alive & (st.rows.w > 0)
    if world == "act":
        idx = st.rows.actual
    else:
        p = st.rows.w * m
        idx = int(r.choice(len(p), p=p / p.sum()))
    j = st.j
    K = cfg.K
    ro = st.rows.RO[idx].copy()
    fa = st.rows.FA[idx].copy()
    s_ = np.array([o.s for o in cfg.ops])
    f_ = np.array([o.f for o in cfg.ops])
    ro[j:] = (r.random(K - j) < s_[j:]).astype(np.int8)
    fa[j:] = (r.random(K - j) < f_[j:]).astype(np.int8)
    rows = Rows(st.rows.Z[idx][None].copy(), ro[None], fa[None], np.ones(1), actual=0)
    s1 = State(rows, np.ones(1, bool), st.correct[idx][None].copy(), st.dup[idx][None].copy(), st.nv, st.kf, j)
    first = {"done": False}

    def choose(dtype, c, s):
        if not first["done"]:
            first["done"] = True
            return action
        pr = pol.probs(dtype, c)
        return int(r.choice(len(pr), p=pr))
    fn = _op_ok if rec["d"] == "V1" else _op_timeout
    end = fn(cfg, s1, j, choose, "leaf")
    return float(returns(cfg, end)["R"][0])


# --------------------------------------------------------------------------------------------- evaluation
def eval_policy(pol: Policy, cfg: Config, rng: np.random.Generator, n_mc: int = 3000) -> dict[str, Any]:
    J = policy_value(cfg, pol.probs)
    Jstar = oracle_value(cfg)
    voi = root_voi(cfg)
    vprob = []
    for j, o in enumerate(cfg.ops):
        c = {"op": o, "kf": False, "c": cfg.c, "j": j, "cfg": cfg}
        vprob.append(float(pol.probs("V1", c)[1]))
    prior = prior_of(cfg)
    tot = defaultdict(float)
    n_ver = n_ver_fault = n_silent = n_silent_ver = n_silent_fixed = n_uninf = 0
    safe_v = safe_n = risky_v = risky_n = 0
    for _ in range(n_mc):
        z = sample_world(cfg, rng, prior)
        ep = sample_episode(cfg, rng, z, prior, lambda d, c, s, r: pol.sample(d, c, s, r))
        for k in ("R", "U", "cost", "fail", "dupl"):
            tot[k] += ep[k]
        tot["succ"] += ep["succ"]
        tot["dup_any"] += ep["dupl"] > 0
        tot["nv"] += ep["nv"]
        fin = ep["state"].correct[ep["state"].rows.actual]
        for j, zz in enumerate(z):
            if zz == 1:
                n_silent += 1
                n_silent_fixed += bool(fin[j])
        for rec in ep["log"]:
            if rec["d"] == "V1":
                safe = cfg.ops[rec["j"]].p == 0
                if safe:
                    safe_n += 1
                    safe_v += rec["a"]
                else:
                    risky_n += 1
                    risky_v += rec["a"]
                if rec["a"] == 1:
                    n_ver += 1
                    n_ver_fault += z[rec["j"]] == 1
                    if z[rec["j"]] == 1:
                        n_silent_ver += 1
            if rec["a"] == 1 and rec["d"] in ("V1", "V2", "V2c") or (rec["d"] == "T" and rec["a"] == 1):
                n_uninf += not rec["info"]
    n = n_mc
    out = {"J": J, "Jstar": Jstar, "regret": Jstar - J, "verify_prob": vprob, "voi": [v["voi"] for v in voi],
           **{f"mc_{k}": tot[k] / n for k in ("R", "U", "cost", "fail", "dupl", "succ", "dup_any", "nv")},
           "precision": n_ver_fault / n_ver if n_ver else None, "recall": n_silent_ver / n_silent if n_silent else None,
           "uninformative_read_frac": n_uninf / tot["nv"] if tot["nv"] else None,
           "safe_verify_rate": safe_v / safe_n if safe_n else None,
           "risky_verify_rate": risky_v / risky_n if risky_n else None,
           "recovery_given_silent": n_silent_fixed / n_silent if n_silent else None,
           "cost_per_completed": tot["cost"] / tot["succ"] if tot["succ"] else None}
    return out


# --------------------------------------------------------------------------------------------- training
def _adv_seq(est: str, g: list[dict], key: str = "R") -> np.ndarray:
    R = np.array([e[key] for e in g])
    mu = R.mean()
    if est in ("grpo", "awrb", "lagr_grpo"):
        return (R - mu) / (R.std(ddof=1) + 1e-4)
    if est in ("nostd", "awrbn", "lagr", "batchstd", "ema", "ctxema") or est in DEC:
        return R - mu
    if est == "rloo":
        G = len(R)
        return R - (R.sum() - R) / (G - 1)
    raise ValueError(est)


def train(spec: dict[str, Any]) -> dict[str, Any]:
    """spec keys: tasks (list[Config]) | task_fn, est, cond ('A' shared / 'B' resampled), steps, groups, G, lr,
    seed, rho0, mode ('tab'|'feat'), prior ('true'|'est'|'mis:k'|'flat:x'), fork_k, budget, opt ('adam'|'sgd'),
    eval_tasks (list[Config]), eval_every, n_mc."""
    rng = np.random.default_rng(spec["seed"])
    est, cond = spec["est"], spec.get("cond", "A")
    tasks: list[Config] = spec["tasks"]
    n_types = 1 + max(o.tid for t in tasks + spec.get("eval_tasks", []) for o in t.ops)
    pol = Policy(n_types, spec.get("mode", "tab"), spec.get("rho0", 0.3))
    opt = Adam(pol.theta, spec.get("lr", 0.05)) if spec.get("opt", "adam") == "adam" else None
    G, groups, steps = spec.get("G", 8), spec.get("groups", 4), spec.get("steps", 300)
    fork_k = spec.get("fork_k", 4)
    prior_spec = spec.get("prior", "true")
    phat_n = defaultdict(lambda: [1.0, 1.0])  # Beta(1,1) counts per op type from revealed V1 reads
    Vhat: dict = {}
    Qhat: dict = {}
    lam = spec.get("lam0", 0.0)
    ema_g: float | None = None
    ema_ctx: dict[str, float] = {}
    episodes = 0
    curve = []
    for step in range(steps):
        batch: list[list[dict]] = []
        for _ in range(groups):
            cfg = tasks[int(rng.integers(len(tasks)))]
            tp = prior_of(cfg)
            if prior_spec == "est":
                ac = assumed_cfg(cfg, "est", {k: v[0] / (v[0] + v[1]) for k, v in phat_n.items()})
                ap = fault_plans(ac)
            elif prior_spec == "true":
                ap = tp
            else:
                ap = prior_of(assumed_cfg(cfg, prior_spec))
            z0 = sample_world(cfg, rng, tp)
            grp = []
            for _ in range(G):
                z = z0 if cond == "A" else sample_world(cfg, rng, tp)
                ep = sample_episode(cfg, rng, z, ap, lambda d, c, s, r: pol.sample(d, c, s, r))
                ep["cfg"] = cfg
                grp.append(ep)
                episodes += 1
            batch.append(grp)
        # empirical-Bayes fault-rate counts from revealed reads (observable outcomes only)
        for grp in batch:
            for ep in grp:
                for rec in ep["log"]:
                    if rec["d"] == "V1" and rec["a"] == 1:
                        seen = bool(ep["z"][rec["j"]] == 1)  # what the read showed (f = 0 in 'est' runs)
                        phat_n[rec["tid"]][0 if seen else 1] += 1
        grad = {d: np.zeros_like(t) for d, t in pol.theta.items()}
        n_dec = sum(len(ep["log"]) for grp in batch for ep in grp)
        # sequence-level advantages
        if est == "batchstd":
            allR = np.array([e["R"] for grp in batch for e in grp])
            sd = allR.std(ddof=1) + 1e-4
            advs = [_adv_seq("nostd", grp) / sd for grp in batch]
        elif est == "gdpo":
            comps = (("U", 1.0), ("cost", -1.0), ("fail", -1.0), ("dupl", -1.0))
            raw = []
            for grp in batch:
                tot = np.zeros(len(grp))
                for k, sign in comps:
                    x = np.array([e[k] for e in grp])
                    tot += sign * (x - x.mean()) / (x.std(ddof=1) + 1e-4)
                raw.append(tot)
            allv = np.concatenate(raw)
            advs = [(r - allv.mean()) / (allv.std(ddof=1) + 1e-4) for r in raw]
        elif est in ("awrb", "awrbn"):
            advs = [_adv_seq(est, grp, "rbar") for grp in batch]
        elif est in ("lagr", "lagr_grpo"):
            for grp in batch:
                for e in grp:
                    e["Rl"] = e["U"] - e["fail"] - e["dupl"] - lam * e["cost"]
            advs = [_adv_seq("nostd" if est == "lagr" else "grpo", grp, "Rl") for grp in batch]
        elif est in ("ema", "ctxema"):
            stds = [float(np.std([e["R"] for e in grp], ddof=1)) for grp in batch]
            if est == "ema":
                div = max(0.02, ema_g if ema_g is not None else float(np.mean(stds)))
                advs = [_adv_seq("nostd", grp) / div for grp in batch]
                ema_g = (ema_g if ema_g is not None else float(np.mean(stds))) * 0.9 + 0.1 * float(np.mean(stds))
            else:
                advs = []
                for grp, sd in zip(batch, stds):
                    tag = grp[0]["cfg"].tag
                    div = max(0.02, ema_ctx.get(tag, sd))
                    advs.append(_adv_seq("nostd", grp) / div)
                for grp, sd in zip(batch, stds):
                    tag = grp[0]["cfg"].tag
                    ema_ctx[tag] = ema_ctx.get(tag, sd) * 0.9 + 0.1 * sd
        else:
            advs = [_adv_seq(est, grp) for grp in batch]
        for grp, A in zip(batch, advs):
            for ep, a in zip(grp, A):
                for rec in ep["log"]:
                    d, g = pol.grad_logp(rec)
                    if est in DEC and rec["d"] in (VERIFY_D if est.startswith("fork") or est == "critic_q" else ()):
                        continue  # handled below by the decision-level term
                    if est in ("igc", "cvc") and rec["d"] in ("V1", "V2", "V2c") and not rec["info"]:
                        continue
                    if est == "cvc" and rec["d"] in VERIFY_D:
                        continue
                    if est == "critic_v":
                        key = (rec["d"], rec["tid"], rec["kf"], rec["j"], ep["cfg"].tag)
                        b = Vhat.get(key, ep["R"])
                        grad[d] += (ep["R"] - b) * g
                        continue
                    grad[d] += a * g
        # decision-level terms
        if est in DEC:
            for grp in batch:
                for ep in grp:
                    cfg = ep["cfg"]
                    for rec in ep["log"]:
                        if est in ("igc", "cvc") and rec["d"] in ("V1", "V2", "V2c") and not rec["info"]:
                            q = np.array([0.0, -cfg.c])
                            d, g = pol.grad_allaction(rec, q)
                            grad[d] += g
                        elif est == "critic_q" and rec["d"] in VERIFY_D:
                            key = (rec["d"], rec["tid"], rec["kf"], rec["j"], cfg.tag)
                            q = np.array([Qhat.get((key, a), ep["R"]) for a in range(len(rec["pr"]))])
                            d, g = pol.grad_allaction(rec, q)
                            grad[d] += g
                        elif (est.startswith("fork") or est == "cvc") and rec["d"] in VERIFY_D:
                            world = "act" if est == "fork_act" else "bel"
                            nA = len(rec["pr"])
                            Q = np.zeros((fork_k, nA))
                            for k in range(fork_k):
                                sd = int(rng.integers(2**31))
                                for a in range(nA):
                                    Q[k, a] = fork_return(cfg, rec, pol, rng, a, world, sd, ep["z"])
                                    episodes += (cfg.K - rec["j"]) / cfg.K  # a fork is a partial episode (pro rata)
                            q = Q.mean(0)
                            if est == "fork_bel_std":
                                diff = Q - Q.mean(1, keepdims=True)
                                q = q / (diff.std() + 1e-4)
                            d, g = pol.grad_allaction(rec, q)
                            grad[d] += g
        for d in grad:
            grad[d] /= max(n_dec, 1)
        if opt is not None:
            opt.step(pol.theta, grad)
        else:
            for d in grad:
                g = grad[d].copy()
                g[0] = 0
                pol.theta[d] += spec.get("lr", 0.05) * g
        # critic updates (after the policy step: the baseline at this step used only earlier data)
        if est in ("critic_v", "critic_q"):
            for grp in batch:
                for ep in grp:
                    for rec in ep["log"]:
                        key = (rec["d"], rec["tid"], rec["kf"], rec["j"], ep["cfg"].tag)
                        if est == "critic_v":
                            Vhat[key] = Vhat.get(key, ep["R"]) * 0.95 + 0.05 * ep["R"]
                        else:
                            kk = (key, rec["a"])
                            Qhat[kk] = Qhat.get(kk, ep["R"]) * 0.95 + 0.05 * ep["R"]
        if est in ("lagr", "lagr_grpo"):
            mc = np.mean([e["cost"] for grp in batch for e in grp])
            lam = max(0.0, lam + spec.get("lam_lr", 2.0) * (mc - spec.get("budget", 0.05)))
        if step % spec.get("curve_every", 10) == 0 or step == steps - 1:
            vr = defaultdict(list)
            for grp in batch:
                for ep in grp:
                    for rec in ep["log"]:
                        if rec["d"] == "V1":
                            vr[rec["tid"]].append(rec["a"])
            zero = np.mean([np.ptp([e["R"] for e in grp]) < 1e-12 for grp in batch])
            curve.append({"step": step, "R": float(np.mean([e["R"] for grp in batch for e in grp])),
                          "verify_by_type": {int(k): float(np.mean(v)) for k, v in vr.items()},
                          "zero_std_frac": float(zero), "lam": lam})
    ev_rng = np.random.default_rng(spec["seed"] + 10_000)
    evals = {t.tag: eval_policy(pol, t, ev_rng, spec.get("n_mc", 2000)) for t in spec.get("eval_tasks", tasks)}
    return {"spec": {k: v for k, v in spec.items() if k not in ("tasks", "eval_tasks")}, "episodes_used": episodes,
            "curve": curve, "eval": evals, "theta": {k: v.tolist() for k, v in pol.theta.items()},
            "phat": {int(k): v[0] / (v[0] + v[1]) for k, v in phat_n.items()}}
