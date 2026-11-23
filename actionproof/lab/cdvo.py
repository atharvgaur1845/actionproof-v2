"""Counterfactual decision-value estimators (CDVO family) and extra baselines on the v2 tabular testbed.

The v2 modules (actionproof/ad2/tabular/*) are registered and frozen; this module imports them and adds estimators.
For every estimator that v2 already had, `train` reproduces v2's `train` exactly (same random stream; tested in
tests/lab/test_cdvo.py), so v2 raw results remain comparable.

New estimators (registered in PREREGISTRATION_LAB1.md):
  igc_v1        igc gating only the first read after a write (the artefact-free gate of v2 TB7)
  ppo           per-decision advantage R - Vhat(observable key) (tabular EMA critic, as critic_v), clipped surrogate,
                4 epochs per batch, clip 0.2, no advantage normalisation  (PPO, Schulman et al. 2017)
  ppo_norm      ppo with per-batch advantage normalisation (the usual PPO implementation detail)
  cdvo_fixed    paired actual-world forks (forced first action, on-policy continuation, common random numbers) at a
                random fraction b of verification decisions; all other decisions: sequence nostd term
  cdvo_adapt    the same fork budget (round(b * #verification decisions) per step), allocated to the decisions whose
                observable context has the least certain sign of the estimated verify-vs-continue difference, from
                forks of *earlier* steps only (rollout allocation as in RSPI, Dimitrakakis and Lagoudakis 2008)
  cdvo_adapt_critic  cdvo_adapt; unforked decisions in confidently-signed contexts use the stored fork mean as a
                learned history-conditioned critic (all-action gradient), others the sequence term
  cdvo_ia       igc_v1 on uninformative first reads (exact -c), cdvo_adapt allocation among informative decisions
  fork_act_nocrn  fork_act with independent random streams for the two branches (CRN ablation)
  indep_world_std  (R - mean_g) / (std of n_ind extra rollouts in the *same* hidden world + 1e-4): the hidden-state
                conditional scale without sharing samples with the gradient (mechanistic intervention)
  indep_task_std   the same, extra rollouts in fresh worlds of the same task: a scale independent of the hidden world
Budget: spec["episode_budget"] stops training once episodes_used >= budget (forks charged pro rata, as in v2).
"""
from __future__ import annotations

from collections import defaultdict, deque
from typing import Any

import numpy as np

import actionproof.ad2.tabular.policy as P
from actionproof.ad2.tabular.env import fault_plans, sample_episode, sample_world
from actionproof.ad2.tabular.policy import Adam, Policy
from actionproof.ad2.tabular.train import DEC, VERIFY_D, _adv_seq, assumed_cfg, eval_policy, fork_return, prior_of

FORKY = {"fork_act", "fork_bel", "fork_bel_std", "cvc", "cdvo_fixed", "cdvo_adapt", "cdvo_adapt_critic", "cdvo_ia",
         "fork_act_nocrn"}
NEW_SEQ = {"igc_v1", "ppo", "ppo_norm", "indep_world_std", "indep_task_std"}
ALL_DEC = DEC | {"igc_v1", "cdvo_fixed", "cdvo_adapt", "cdvo_adapt_critic", "cdvo_ia", "fork_act_nocrn"}
GATED_V1 = {"igc_v1", "cdvo_ia"}


def _contrast(d: str, q: np.ndarray) -> float:
    """Verify-vs-continue difference: V1 Q(verify) - Q(skip); T Q(verify) - max(Q(proceed), Q(retry))."""
    return float(q[1] - q[0]) if d == "V1" else float(q[1] - max(q[0], q[2]))


def _key(rec: dict, cfg) -> tuple:
    return (rec["d"], rec["tid"], rec["kf"], rec["j"], cfg.tag)


def _informative_v1_only(orig):
    def gate(dtype, j, st):
        if dtype in ("V2", "V2c"):
            return True
        return orig(dtype, j, st)
    return gate


SUPPORTED = {"grpo", "nostd", "rloo", "batchstd", "gdpo", "awrb", "awrbn", "critic_v", "critic_q", "igc", "cvc",
             "fork_act", "fork_bel", "fork_bel_std"} | NEW_SEQ | ALL_DEC


def train(spec: dict[str, Any]) -> dict[str, Any]:
    est = spec["est"]
    if est not in SUPPORTED:
        raise ValueError(f"{est} is not supported by actionproof.lab.cdvo (use actionproof.ad2.tabular.train)")
    orig_inf = P.informative
    if est in GATED_V1:
        P.informative = _informative_v1_only(orig_inf)
    try:
        return _train(spec)
    finally:
        P.informative = orig_inf


def _train(spec: dict[str, Any]) -> dict[str, Any]:
    rng = np.random.default_rng(spec["seed"])
    est, cond = spec["est"], spec.get("cond", "A")
    tasks = spec["tasks"]
    n_types = 1 + max(o.tid for t in tasks + spec.get("eval_tasks", []) for o in t.ops)
    pol = Policy(n_types, spec.get("mode", "tab"), spec.get("rho0", 0.3))
    if spec.get("theta0") is not None:  # fixed-parameter studies (scripts/lab/grad_snr.py)
        pol.theta = {k: np.array(v, dtype=float) for k, v in spec["theta0"].items()}
    opt = Adam(pol.theta, spec.get("lr", 0.05)) if spec.get("opt", "adam") == "adam" else None
    G, groups, steps = spec.get("G", 8), spec.get("groups", 4), spec.get("steps", 300)
    budget = spec.get("episode_budget")
    if budget is not None:
        steps = 10 ** 6
    fork_k = spec.get("fork_k", 1)
    b = spec.get("branch_frac", 0.25)
    n_ind = spec.get("n_ind", 16)
    prior_spec = spec.get("prior", "true")
    phat_n = defaultdict(lambda: [1.0, 1.0])
    Vhat: dict = {}
    Qhat: dict = {}
    lam = spec.get("lam0", 0.0)
    hist: dict = defaultdict(lambda: deque(maxlen=spec.get("hist_window", 20)))  # key -> past fork contrasts
    qmean: dict = {}  # key -> EMA of fork Q vectors (cdvo_adapt_critic)
    episodes = 0.0
    n_forks = 0
    n_updates = 0
    is_dec = est in ALL_DEC
    curve = []
    for step in range(steps):
        if budget is not None and episodes >= budget:
            break
        batch = []
        for _ in range(groups):
            cfg = tasks[int(rng.integers(len(tasks)))]
            tp = prior_of(cfg)
            if prior_spec == "est":
                ap = fault_plans(assumed_cfg(cfg, "est", {k: v[0] / (v[0] + v[1]) for k, v in phat_n.items()}))
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
            if est in ("indep_world_std", "indep_task_std"):
                xs = []
                for _ in range(n_ind):
                    zz = z0 if est == "indep_world_std" else sample_world(cfg, rng, tp)
                    xs.append(sample_episode(cfg, rng, zz, ap, lambda d, c, s, r: pol.sample(d, c, s, r))["R"])
                    episodes += 1
                grp[0]["_ind_sd"] = float(np.std(xs, ddof=1))
            batch.append(grp)
        for grp in batch:
            for ep in grp:
                for rec in ep["log"]:
                    if rec["d"] == "V1" and rec["a"] == 1:
                        phat_n[rec["tid"]][0 if bool(ep["z"][rec["j"]] == 1) else 1] += 1
        grad = {d: np.zeros_like(t) for d, t in pol.theta.items()}
        n_dec = sum(len(ep["log"]) for grp in batch for ep in grp)
        # ---- choose which verification decisions are forked (CDVO variants)
        fork_set: set = set()
        if est in ("cdvo_fixed", "cdvo_adapt", "cdvo_adapt_critic", "cdvo_ia"):
            cands = []
            for gi, grp in enumerate(batch):
                for ei, ep in enumerate(grp):
                    for ri, rec in enumerate(ep["log"]):
                        if rec["d"] not in VERIFY_D:
                            continue
                        if est == "cdvo_ia" and rec["d"] == "V1" and not rec["info"]:
                            continue
                        cands.append((gi, ei, ri))
            if est == "cdvo_fixed":
                fork_set = {c for c in cands if rng.random() < b}
            else:
                nf = int(round(b * len(cands)))
                if nf and cands:
                    scores = []
                    for (gi, ei, ri) in cands:
                        ep = batch[gi][ei]
                        h = hist[_key(ep["log"][ri], ep["cfg"])]
                        if len(h) < 2:
                            z = 0.0
                        else:
                            a = np.array(h)
                            z = abs(a.mean()) / (a.std(ddof=1) / np.sqrt(len(a)) + 1e-6)
                        scores.append(z + 1e-9 * rng.random())
                    order = np.argsort(scores)
                    fork_set = {cands[i] for i in order[:nf]}
        # ---- sequence-level advantages
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
        elif est in ("indep_world_std", "indep_task_std"):
            advs = [_adv_seq("nostd", grp) / (grp[0]["_ind_sd"] + 1e-4) for grp in batch]
        elif est in ("igc_v1", "ppo", "ppo_norm") or est in ALL_DEC:
            advs = [_adv_seq("nostd", grp) for grp in batch]
        else:
            advs = [_adv_seq(est, grp) for grp in batch]
        if est in ("ppo", "ppo_norm"):
            _ppo_update(pol, opt, batch, Vhat, n_dec, est == "ppo_norm", spec)
            n_updates += spec.get("ppo_epochs", 4)
        else:
            gated = est in ("igc", "cvc") or est in GATED_V1
            for gi, (grp, A) in enumerate(zip(batch, advs)):
                for ei, (ep, a) in enumerate(zip(grp, A)):
                    for ri, rec in enumerate(ep["log"]):
                        d, g = pol.grad_logp(rec)
                        if rec["d"] in VERIFY_D and (est in ("fork_act", "fork_bel", "fork_bel_std", "fork_act_nocrn",
                                                             "critic_q")):
                            continue
                        if gated and rec["d"] in ("V1", "V2", "V2c") and not rec["info"]:
                            continue
                        if est == "cvc" and rec["d"] in VERIFY_D:
                            continue
                        if (gi, ei, ri) in fork_set:
                            continue
                        if est == "cdvo_adapt_critic" and rec["d"] in VERIFY_D:
                            k = _key(rec, ep["cfg"])
                            h = hist[k]
                            if len(h) >= 3 and k in qmean:
                                arr = np.array(h)
                                if abs(arr.mean()) / (arr.std(ddof=1) / np.sqrt(len(arr)) + 1e-6) >= 2.0:
                                    continue  # handled by the learned-critic term below
                        if est == "critic_v":
                            key = (rec["d"], rec["tid"], rec["kf"], rec["j"], ep["cfg"].tag)
                            grad[d] += (ep["R"] - Vhat.get(key, ep["R"])) * g
                            continue
                        grad[d] += a * g
            if is_dec:
                new_hist = []
                for gi, grp in enumerate(batch):
                    for ei, ep in enumerate(grp):
                        cfg = ep["cfg"]
                        for ri, rec in enumerate(ep["log"]):
                            if gated and rec["d"] in ("V1", "V2", "V2c") and not rec["info"]:
                                q = np.array([0.0, -cfg.c])
                                d, g = pol.grad_allaction(rec, q)
                                grad[d] += g
                            elif est == "critic_q" and rec["d"] in VERIFY_D:
                                key = (rec["d"], rec["tid"], rec["kf"], rec["j"], cfg.tag)
                                q = np.array([Qhat.get((key, a), ep["R"]) for a in range(len(rec["pr"]))])
                                d, g = pol.grad_allaction(rec, q)
                                grad[d] += g
                            elif rec["d"] in VERIFY_D and (
                                    est in ("fork_act", "fork_bel", "fork_bel_std", "cvc", "fork_act_nocrn")
                                    or (gi, ei, ri) in fork_set):
                                world = "bel" if est in ("fork_bel", "fork_bel_std", "cvc") else "act"
                                nA = len(rec["pr"])
                                Q = np.zeros((fork_k, nA))
                                for k in range(fork_k):
                                    sd = int(rng.integers(2 ** 31))
                                    for a in range(nA):
                                        s_a = sd if est != "fork_act_nocrn" else int(rng.integers(2 ** 31))
                                        Q[k, a] = fork_return(cfg, rec, pol, rng, a, world, s_a, ep["z"])
                                        episodes += (cfg.K - rec["j"]) / cfg.K
                                    n_forks += 1
                                q = Q.mean(0)
                                if est == "fork_bel_std":
                                    diff = Q - Q.mean(1, keepdims=True)
                                    q = q / (diff.std() + 1e-4)
                                d, g = pol.grad_allaction(rec, q)
                                grad[d] += g
                                if est.startswith("cdvo"):
                                    new_hist.append((_key(rec, cfg), _contrast(rec["d"], q), q))
                            elif est == "cdvo_adapt_critic" and rec["d"] in VERIFY_D:
                                k = _key(rec, cfg)
                                h = hist[k]
                                if len(h) >= 3 and k in qmean:
                                    arr = np.array(h)
                                    if abs(arr.mean()) / (arr.std(ddof=1) / np.sqrt(len(arr)) + 1e-6) >= 2.0:
                                        d, g = pol.grad_allaction(rec, qmean[k])
                                        grad[d] += g
                for k, dlt, q in new_hist:  # statistics are used from the next step on
                    hist[k].append(dlt)
                    qmean[k] = q.copy() if k not in qmean else 0.8 * qmean[k] + 0.2 * q
            for d in grad:
                grad[d] /= max(n_dec, 1)
            if opt is not None:
                opt.step(pol.theta, grad)
            else:
                for d in grad:
                    g = grad[d].copy()
                    g[0] = 0
                    pol.theta[d] += spec.get("lr", 0.05) * g
            n_updates += 1
        if est in ("critic_v", "critic_q", "ppo", "ppo_norm"):
            for grp in batch:
                for ep in grp:
                    for rec in ep["log"]:
                        key = (rec["d"], rec["tid"], rec["kf"], rec["j"], ep["cfg"].tag)
                        if est == "critic_q":
                            kk = (key, rec["a"])
                            Qhat[kk] = Qhat.get(kk, ep["R"]) * 0.95 + 0.05 * ep["R"]
                        else:
                            Vhat[key] = Vhat.get(key, ep["R"]) * 0.95 + 0.05 * ep["R"]
        if step % spec.get("curve_every", 10) == 0:
            curve.append({"step": step, "episodes": episodes,
                          "R": float(np.mean([e["R"] for grp in batch for e in grp]))})
    ev_rng = np.random.default_rng(spec["seed"] + 10_000)
    evals = {t.tag: eval_policy(pol, t, ev_rng, spec.get("n_mc", 2000)) for t in spec.get("eval_tasks", tasks)}
    return {"spec": {k: v for k, v in spec.items() if k not in ("tasks", "eval_tasks", "theta0")}, "episodes_used": episodes,
            "steps_done": step + 1 if budget is None else step, "n_forks": n_forks, "n_updates": n_updates,
            "curve": curve, "eval": evals, "theta": {k: v.tolist() for k, v in pol.theta.items()}}


def _ppo_update(pol: Policy, opt: Adam, batch, Vhat: dict, n_dec: int, normalise: bool, spec: dict) -> None:
    recs, advs = [], []
    for grp in batch:
        for ep in grp:
            for rec in ep["log"]:
                key = (rec["d"], rec["tid"], rec["kf"], rec["j"], ep["cfg"].tag)
                recs.append(rec)
                advs.append(ep["R"] - Vhat.get(key, ep["R"]))
    A = np.array(advs)
    if normalise and len(A) > 1:
        A = (A - A.mean()) / (A.std(ddof=1) + 1e-8)
    clip = spec.get("ppo_clip", 0.2)
    for _ in range(spec.get("ppo_epochs", 4)):
        grad = {d: np.zeros_like(t) for d, t in pol.theta.items()}
        for rec, a in zip(recs, A):
            pr_new = pol.probs_phi(rec["d"], rec["phi"])
            ratio = pr_new[rec["a"]] / rec["pr"][rec["a"]]
            if (a > 0 and ratio > 1 + clip) or (a < 0 and ratio < 1 - clip):
                continue
            e = np.zeros(len(pr_new))
            e[rec["a"]] = 1.0
            grad[rec["d"]] += a * ratio * np.outer(e - pr_new, rec["phi"])
        for d in grad:
            grad[d] /= max(n_dec, 1)
        opt.step(pol.theta, grad)
