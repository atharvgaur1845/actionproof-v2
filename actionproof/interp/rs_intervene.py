"""M3 (docs/INTERP_PLAN.md): ablate the model's own A2 discrepancy-probe direction at its probe layer over the final
tool-result message; dose-response α ∈ {0 (sham), 0.5, 1, 2}; 5 random unit directions as controls; collateral check
on benign first-turn prefixes. Additive steering (+β·σ·d) in a second model. Single-decision readout only.

    python -m actionproof.interp.rs_intervene --ablate-tag outcome_s0 --ablate-model runs/grpo_tiny_v2_outcome_s0/merged \
        --steer-tag ack_outcome_s0 --steer-model runs/grpo_tiny_v2_ack_outcome_s0/merged --out results/interp/rs/m3.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from actionproof.interp.probe import task_split
from actionproof.interp.rs_extract import GROUPS, Runner


def group_probs(p):
    return {k: sum(p[t] for t in ts) for k, ts in GROUPS.items()}


def eval_set(R, rows, fn=None, layer=None):
    R.hook = (layer, fn) if fn is not None else None
    out = []
    for r in rows:
        _, p = R.run(r["conv"], want_tools=True)
        out.append(group_probs(p))
    R.hook = None
    return out


def ablate(d, alpha):
    def fn(hs, s0):
        dd = d.to(hs.dtype)
        seg = hs[:, s0:]
        hs = hs.clone()
        hs[:, s0:] = seg - alpha * (seg @ dd)[..., None] * dd
        return hs
    return fn


def add(d, amount):
    def fn(hs, s0):
        hs = hs.clone()
        hs[:, s0:] = hs[:, s0:] + amount * d.to(hs.dtype)
        return hs
    return fn


def summary(base, pert, rows):
    out = {}
    for lab in (1, 0):
        idx = [i for i, r in enumerate(rows) if r["label"] == lab]
        for k in ("recover", "verify", "proceed"):
            out[f"d_{k}_label{lab}"] = round(float(np.mean([pert[i][k] - base[i][k] for i in idx])), 4)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefixes", default="results/interp/rs/prefixes.jsonl")
    ap.add_argument("--directions", default="results/interp/rs/directions.json")
    ap.add_argument("--ablate-tag", required=True)
    ap.add_argument("--ablate-model", required=True)
    ap.add_argument("--steer-tag", required=True)
    ap.add_argument("--steer-model", required=True)
    ap.add_argument("--benign", default="results/eval/r3/grpo_tiny_v2_outcome_s0__neutral__test_iid.jsonl")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    import torch
    D = json.loads(Path(a.directions).read_text())
    rows = [r for l in open(a.prefixes) if (r := json.loads(l))["kind"] == "A2" and task_split(r["task_id"]) == "test"]
    benign = [{"conv": json.loads(l)["conversation"][:2], "next": json.loads(l)["trace"][0]["tool"]}
              for l in list(open(a.benign))[:100]]
    res = {"n_test": len(rows), "n_label1": sum(r["label"] for r in rows)}

    R = Runner(a.ablate_model)
    dir_ = D[a.ablate_tag]
    L, d = dir_["layer"], torch.tensor(dir_["d"], device="cuda")
    base = eval_set(R, rows)
    res["ablate"] = {"tag": a.ablate_tag, "layer": L, "base": {
        f"P_{k}_label{lab}": round(float(np.mean([b[k] for b, r in zip(base, rows) if r["label"] == lab])), 4)
        for k in ("recover", "verify", "proceed") for lab in (1, 0)}}
    for alpha in (0.0, 0.5, 1.0, 2.0):
        res["ablate"][f"probe_alpha{alpha}"] = summary(base, eval_set(R, rows, ablate(d, alpha), L), rows)
    g = torch.Generator(device="cpu").manual_seed(0)
    for k in range(5):
        rd = torch.randn(len(dir_["d"]), generator=g); rd = (rd / rd.norm()).to("cuda")
        for alpha in (1.0, 2.0):
            res["ablate"][f"random{k}_alpha{alpha}"] = summary(base, eval_set(R, rows, ablate(rd, alpha), L), rows)
    # collateral: benign first-turn prefixes, probability of the actual next tool (renormalised over candidates)
    def p_next(fn):
        R.hook = (L, fn) if fn is not None else None
        v = []
        for b in benign:
            _, p = R.run(b["conv"], want_tools=True)
            v.append(p.get(b["next"], 0.0))
        R.hook = None
        return v
    b0, b1 = p_next(None), p_next(ablate(d, 1.0))
    res["ablate"]["collateral_mean_abs_dP_next_alpha1"] = round(float(np.mean(np.abs(np.array(b1) - np.array(b0)))), 4)
    del R; torch.cuda.empty_cache()

    R = Runner(a.steer_model)
    dir_ = D[a.steer_tag]
    L, d, sd = dir_["layer"], torch.tensor(dir_["d"], device="cuda"), dir_["proj_std"]
    base = eval_set(R, rows)
    res["steer"] = {"tag": a.steer_tag, "layer": L, "proj_std": sd}
    for beta in (1.0, 2.0, 4.0):
        res["steer"][f"probe_plus{beta}sd"] = summary(base, eval_set(R, rows, add(d, beta * sd), L), rows)
        rd = torch.randn(len(dir_["d"]), generator=g); rd = (rd / rd.norm()).to("cuda")
        res["steer"][f"random_plus{beta}sd"] = summary(base, eval_set(R, rows, add(rd, beta * sd), L), rows)
    Path(a.out).write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
