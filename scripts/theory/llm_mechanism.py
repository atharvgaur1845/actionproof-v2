"""Mechanism diagnostics on the logged AD training rollouts (Tiny Aya Global, cw_train; development, descriptive).

    python scripts/theory/llm_mechanism.py --rollouts /scratch/pabitra/actionproof/results/ad/rollouts \
        --out results/research/llm_mechanism.json

Every rollout record (actionproof/ad/rollout.py, AP_ROLLOUT_LOG) carries step, task (with meta.faults), score.r_v,
n_verify and ad.r_bar. A TRL group is the 8 rollouts of one prompt in one step (num_generations = 8; 4 prompts per
step). For every logged run this script

  1. checks group construction: 8 rollouts per (step, task) and one fault plan per group (shared hidden state);
  2. rebuilds the arm's training reward R (r_v or r_bar, minus c * n_verify; c from the run name) and recomputes,
     on the SAME rollouts, the GRPO advantage (Bessel std + 1e-4, TRL 1.7.1) and the centred (Dr. GRPO) advantage;
  3. computes the read-propensity update proxy u_g = sum_i A_i (x_i - mean x), x_i = n_verify_i, per group, split by
     world class and step window, for both estimators (counterfactual estimator, data fixed);
  4. estimates within-group effects beta_z = mean R(readers) - mean R(non-readers) per world class from the first
     20 steps only, and from them the vote sum_z w_z sign(beta_z) and delta_q = sum_z w_z beta_z (the theory's
     predicted direction for GRPO and Dr. GRPO), with a group bootstrap;
  5. counterfactual cost sweep: recomputes U_grpo(c) and U_centred(c) on the same rollouts for c in COSTS
     (Theorem 2 predicts GRPO's clean-group contribution is exactly invariant to c);
  6. reader homogeneity: whether verifying rollouts' returns depend on the hidden state (Corollary 3.3 condition);
  7. read share per 10-step window (checkpoint-level read propensity, sampled rollouts).
"""
from __future__ import annotations

import argparse
import gzip
import json
import math
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

COSTS = (0.0, 0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0)
SILENT = ("phantom", "partial", "dup_delivery")


def arm_info(name: str) -> dict:
    field = "r_bar" if name.startswith("ad_awrb") else "r_v"
    c = 0.30 if "c30" in name else 0.05
    scale = "group" if (name.startswith("ad_grpo") or re.match(r"ad_awrb(c30)?_s", name)) else "none"
    if "ema" in name:
        scale = "ema"
    return {"field": field, "c": c, "train_scale": scale}


def wclass(fc: str) -> str:
    if fc == "cw:clean":
        return "clean"
    kind = fc.split(":")[1]
    return ("silent:" if kind in SILENT else "visible:") + kind


def load(path: Path) -> list[dict]:
    out = []
    with gzip.open(path, "rt") as f:
        for line in f:
            r = json.loads(line)
            out.append({"step": r["step"], "task": r["task_id"], "faults": json.dumps(r["task"]["meta"].get("faults")),
                        "cls": wclass(r["fault_class"]), "fc": r["fault_class"], "r_v": float(r["score"]["r_v"]),
                        "r_bar": (r.get("ad") or {}).get("r_bar"), "reads": int(r["n_verify"]),
                        "vsc": int(r["score"]["vsc"]), "fs": bool(r["score"]["false_success"])})
    return out


def groups_of(rows: list[dict]) -> list[list[dict]]:
    g = defaultdict(list)
    for r in rows:
        g[(r["step"], r["task"])].append(r)
    return list(g.values())


def adv(R: np.ndarray, est: str) -> np.ndarray:
    d = R - R.mean()
    if est == "grpo":
        return d / (R.std(ddof=1) + 1e-4)
    return d


def base(r: dict, field: str) -> float:
    v = r[field]
    return float(r["r_v"] if v is None else v)


def window(step: int) -> str:
    for lo, hi in ((0, 20), (20, 50), (50, 100), (100, 150), (150, 200)):
        if lo <= step < hi:
            return f"{lo}-{hi}"
    return "200+"


def analyse(rows: list[dict], info: dict, rng: np.random.Generator) -> dict:
    G = groups_of(rows)
    sizes = [len(g) for g in G]
    shared = sum(len({r["faults"] for r in g}) == 1 for g in G)
    res = {"rollouts": len(rows), "groups": len(G), "group_sizes": sorted(set(sizes)),
           "groups_with_one_fault_plan": shared, "share_groups_shared_state": shared / len(G)}
    field, c = info["field"], info["c"]
    # --- 3. counterfactual-estimator decomposition at the trained cost
    dec = defaultdict(lambda: defaultdict(float))
    nz = defaultdict(lambda: [0, 0])
    for g in G:
        R = np.array([base(r, field) - c * r["reads"] for r in g])
        x = np.array([r["reads"] for r in g], dtype=float)
        cls, w = g[0]["cls"], window(g[0]["step"])
        nz[(cls, w)][1] += 1
        if R.std() < 1e-12:
            nz[(cls, w)][0] += 1
        for est in ("grpo", "centred"):
            u = float((adv(R, est) * (x - x.mean())).sum())
            dec[(est, w)][cls] += u
            dec[(est, "all")][cls] += u
    res["decomposition"] = {f"{e}|{w}": dict(v) for (e, w), v in dec.items()}
    res["zero_variance_share"] = {f"{k[0]}|{k[1]}": v[0] / v[1] for k, v in nz.items()}
    # --- 4. within-group beta by class from steps 0-19
    early = [g for g in G if g[0]["step"] < 20]
    by = defaultdict(list)
    for g in early:
        R = np.array([base(r, field) - c * r["reads"] for r in g])
        x1 = np.array([r["reads"] > 0 for r in g])
        if 0 < x1.sum() < len(g):
            by[g[0]["cls"]].append(float(R[x1].mean() - R[~x1].mean()))
    cnt = defaultdict(int)
    for g in G:
        cnt[g[0]["cls"]] += 1
    tot = sum(cnt.values())
    wts = {k: v / tot for k, v in cnt.items()}

    def summarise(bydict):
        beta = {k: float(np.mean(v)) for k, v in bydict.items() if v}
        vote = sum(wts[k] * np.sign(b) for k, b in beta.items())
        dq = sum(wts[k] * b for k, b in beta.items())
        return beta, float(vote), float(dq)

    beta, vote, dq = summarise(by)
    boots = []
    keys = list(by)
    for _ in range(2000):
        bb = {k: list(rng.choice(by[k], size=len(by[k]))) for k in keys}
        boots.append(summarise(bb)[1:])
    boots = np.array(boots)
    res["early_effects"] = {"beta_by_class": beta, "n_groups_by_class": {k: len(v) for k, v in by.items()},
                            "class_weights": wts, "vote": vote, "delta_q": dq,
                            "vote_ci95": [float(np.quantile(boots[:, 0], 0.025)), float(np.quantile(boots[:, 0], 0.975))],
                            "delta_q_ci95": [float(np.quantile(boots[:, 1], 0.025)),
                                             float(np.quantile(boots[:, 1], 0.975))],
                            "P_vote_negative": float((boots[:, 0] < 0).mean()),
                            "P_delta_q_positive": float((boots[:, 1] > 0).mean())}
    # --- 4b. the same within-group effects per step window (exploratory: is delta_q non-stationary?)
    per_w = {}
    for lo, hi in ((0, 20), (20, 50), (50, 100), (100, 150), (150, 200)):
        bw = defaultdict(list)
        for g in G:
            if not lo <= g[0]["step"] < hi:
                continue
            R = np.array([base(r, field) - c * r["reads"] for r in g])
            x1 = np.array([r["reads"] > 0 for r in g])
            if 0 < x1.sum() < len(g):
                bw[g[0]["cls"]].append(float(R[x1].mean() - R[~x1].mean()))
        if bw:
            bt, vt, dqt = summarise(bw)
            per_w[f"{lo}-{hi}"] = {"vote": vt, "delta_q": dqt, "beta_by_class": bt,
                                   "n_groups_mixed": sum(len(v) for v in bw.values())}
    res["effects_by_window"] = per_w
    # --- 4c. why clean groups vary: reads only (vote regime) or also R_V (other within-state noise)
    cg = {"nonzero_var": 0, "rv_constant": 0, "rv_varies": 0}
    for g in G:
        if g[0]["cls"] != "clean":
            continue
        R = np.array([base(r, field) - c * r["reads"] for r in g])
        if R.std() < 1e-12:
            continue
        cg["nonzero_var"] += 1
        rv = np.array([base(r, field) for r in g])
        cg["rv_constant" if rv.std() < 1e-12 else "rv_varies"] += 1
    res["clean_group_variance_source"] = cg
    # --- 5. counterfactual cost sweep (all steps and early steps)
    sweep = {}
    for label, gs in (("all", G), ("early", early)):
        for cc in COSTS:
            tot_ = {"grpo": 0.0, "centred": 0.0, "grpo_clean": 0.0, "centred_clean": 0.0}
            for g in gs:
                R = np.array([base(r, field) - cc * r["reads"] for r in g])
                x = np.array([r["reads"] for r in g], dtype=float)
                for est in ("grpo", "centred"):
                    u = float((adv(R, est) * (x - x.mean())).sum())
                    tot_[est] += u
                    if g[0]["cls"] == "clean":
                        tot_[est + "_clean"] += u
            sweep[f"{label}|c={cc}"] = tot_
    res["cost_sweep"] = sweep
    # --- 6. reader homogeneity: return of verifying rollouts by class
    rh = defaultdict(list)
    for r in rows:
        if r["reads"] > 0:
            rh[r["cls"]].append(base(r, field))
    res["reader_return_by_class"] = {k: {"n": len(v), "mean_base": float(np.mean(v))} for k, v in rh.items()}
    eq = [0, 0]
    for g in G:
        rd = [base(r, field) for r in g if r["reads"] > 0]
        if len(rd) >= 2:
            eq[1] += 1
            eq[0] += int(max(rd) - min(rd) < 1e-9)
    res["groups_readers_equal_base_return"] = eq
    # --- 7. read share per 10-step window
    share = defaultdict(lambda: [0, 0])
    for r in rows:
        k = r["step"] // 10 * 10
        share[k][1] += 1
        share[k][0] += int(r["reads"] > 0)
    res["read_share_by_10_steps"] = {str(k): v[0] / v[1] for k, v in sorted(share.items())}
    first = [r for r in rows if r["step"] < 20]
    last = [r for r in rows if r["step"] >= 150]
    res["read_share_first20"] = float(np.mean([r["reads"] > 0 for r in first]))
    res["read_share_last50"] = float(np.mean([r["reads"] > 0 for r in last]))
    res["vsc_last50"] = float(np.mean([r["vsc"] for r in last]))
    res["false_success_last50"] = float(np.mean([r["fs"] for r in last]))
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rollouts", default="/scratch/pabitra/actionproof/results/ad/rollouts")
    ap.add_argument("--out", default="results/research/llm_mechanism.json")
    a = ap.parse_args()
    rng = np.random.default_rng(20261124)
    out = {}
    for p in sorted(Path(a.rollouts).glob("ad_*.jsonl.gz")):
        name = p.name.replace(".jsonl.gz", "")
        info = arm_info(name)
        rows = load(p)
        out[name] = {"info": info, **analyse(rows, info, rng)}
        print(name, out[name]["read_share_first20"], out[name]["read_share_last50"],
              out[name]["early_effects"]["vote"], out[name]["early_effects"]["delta_q"], flush=True)
    import hashlib
    out["_inputs"] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(Path(a.rollouts).glob("ad_*.jsonl.gz"))}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=1))
    print("wrote", a.out)


if __name__ == "__main__":
    main()
