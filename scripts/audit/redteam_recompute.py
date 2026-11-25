"""Red-team recomputations (research_audit/RED_TEAM_VERIFICATION.md, OPTIMIZER_AUDIT.md). Descriptive and read-only,
written 2026-11-25 after every result named here was known. Writes results/audit/redteam_recompute.json.

1. Cell-clustered bootstrap (the red-team method, docs/RESEARCH_RED_TEAM_REVIEW.md Appendix 1): cell =
   (fault_class, workflow, first fault tool, start_type); resample cells with replacement, B = 2,000, seed 0; the
   statistic is the seed-averaged per-task VSC difference over all tasks of the resampled cells. Applied to
   S3 (greedy, one episode per task) and P1/P2 (sampled VSC, mean of 4 samples, as the registered P1 unit).
2. Fisher exact (one-sided, hypergeometric) on runs that learned partial repair (partial-class VSC >= 0.40).
3. Training-log scan over every runs/*/ with a trainer_state.json: clip ratio, entropy, zero-std group share,
   vLLM importance-weight masking and the largest sampler/trainer log-prob gap.

    python scripts/audit/redteam_recompute.py
"""
from __future__ import annotations

import glob
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def cell(r):
    t = r["task"]
    m = t.get("meta", {})
    f = (m.get("faults") or [{}])[0]
    return (r.get("fault_class") or m.get("fault_class"), t.get("workflow"), f.get("tool"), r.get("start_type") or "full")


def load_scores(path, sampled):
    by, cells = defaultdict(list), {}
    for r in map(json.loads, Path(path).read_text().splitlines()):
        s = r.get("sample")
        if sampled and (s is None or s < 0):
            continue
        if not sampled and s is not None and s >= 0:
            continue
        by[r["task_id"]].append(r["vsc"])
        cells[r["task_id"]] = cell(r)
    return {k: float(np.mean(v)) for k, v in by.items()}, cells


def arm(files, sampled):
    per, cells = [], {}
    for f in files:
        s, c = load_scores(f, sampled)
        per.append(s); cells.update(c)
    keys = sorted(set.intersection(*(set(p) for p in per)))
    return {k: float(np.mean([p[k] for p in per])) for k in keys}, cells, per


def clustered(x, y, cells, B=2000, seed=0):
    keys = sorted(set(x) & set(y))
    d = {k: x[k] - y[k] for k in keys}
    groups = defaultdict(list)
    for k in keys:
        groups[cells[k]].append(d[k])
    cl = list(groups.values())
    rng = np.random.default_rng(seed)
    sums = np.array([sum(g) for g in cl]); ns = np.array([len(g) for g in cl])
    idx = rng.integers(0, len(cl), size=(B, len(cl)))
    bs = sums[idx].sum(1) / ns[idx].sum(1)
    # task-level bootstrap for comparison
    dv = np.array(list(d.values()))
    tidx = rng.integers(0, len(dv), size=(B, len(dv)))
    tb = dv[tidx].mean(1)
    return {"delta": float(dv.mean()), "n_tasks": len(dv), "n_cells": len(cl),
            "cell_ci95": [float(np.quantile(bs, .025)), float(np.quantile(bs, .975))],
            "task_ci95": [float(np.quantile(tb, .025)), float(np.quantile(tb, .975))],
            "cells_constant_within_x": None}


def fisher_one_sided(a, n1, b, n2):
    """P(X >= a) for X ~ Hypergeometric(total learners a+b, n1, n2): arm 1 has a of n1 learners, arm 2 b of n2."""
    K, N = a + b, n1 + n2
    tot = math.comb(N, K)
    return sum(math.comb(n1, i) * math.comb(n2, K - i) for i in range(a, min(K, n1) + 1)) / tot


def s3_files(tag, seeds, split):
    return [ROOT / f"results/s3/conf/{tag}_s{s}__{split}.jsonl" for s in seeds]


def p1_files(tag):
    d = "results/p2/test" if tag in ("E",) else "results/p2/ref" if tag == "M" else "results/p1/test"
    out = []
    for sp in ("p1_test_full", "p1_test_pp", "p1_test_rec", "p1_test_wf4", "p1_test_combo"):
        out.append([ROOT / f"{d}/{tag}_s{s}__{sp}.jsonl" for s in range(3)])
    return out


def main():
    out = {"s3": {}, "p1": {}, "fisher": {}, "train_logs": {}}
    five = range(5)
    for split in ("s3_conf_full", "s3_conf_rec"):
        rsc, cells, _ = arm(s3_files("s3_rv_mix", five, split), sampled=False)
        for name, tag, seeds in (("RSC-A12RV", "a12_rv", five), ("RSC-CONTROL", "s3_matched", five),
                                 ("RSC-ACK", "s3_ackonly", five)):
            other, c2, _ = arm(s3_files(tag, seeds, split), sampled=False)
            cells.update(c2)
            out["s3"][f"{name}|{split}"] = clustered(rsc, other, cells)
        # partial class only, full episodes
    rsc, cells, rsc_per = arm(s3_files("s3_rv_mix", five, "s3_conf_full"), sampled=False)
    for tag in ("s3_matched", "s3_ackonly", "a12_rv"):
        o, c2, o_per = arm(s3_files(tag, five, "s3_conf_full"), sampled=False)
        cells.update(c2)
        part = {k for k in rsc if cells[k][0] == "partial"}
        out["s3"][f"RSC-{tag}|s3_conf_full|partial"] = clustered({k: rsc[k] for k in part}, {k: o[k] for k in part if k in o}, cells)
        learned = lambda per: [float(np.mean([p[k] for k in part if k in p])) for p in per]
        lr, lo = learned(rsc_per), learned(o_per)
        a, b = sum(v >= 0.40 for v in lr), sum(v >= 0.40 for v in lo)
        out["fisher"][f"RSC vs {tag} (partial-class VSC >= 0.40, s3_conf_full)"] = {
            "rsc_per_run": lr, "other_per_run": lo, "rsc_learners": a, "other_learners": b,
            "p_one_sided": fisher_one_sided(a, 5, b, 5)}

    # P1/P2: sampled VSC, test union
    def p1_arm(tag):
        files_by_split = p1_files(tag)
        per_seed = [dict() for _ in range(3)]; cells = {}
        for files in files_by_split:
            for s, f in enumerate(files):
                sc, c = load_scores(f, sampled=True)
                per_seed[s].update({(f.name.split("__")[1], k): v for k, v in sc.items()})
                cells.update({(f.name.split("__")[1], k): v for k, v in c.items()})
        keys = sorted(set.intersection(*(set(p) for p in per_seed)))
        return {k: float(np.mean([p[k] for p in per_seed])) for k in keys}, cells
    arms = {t: p1_arm(t) for t in ("A", "C", "D", "E", "M")}
    cells = {}
    for _, c in arms.values():
        cells.update(c)
    for x, y in (("C", "A"), ("D", "A"), ("E", "A"), ("D", "C"), ("A", "M"), ("D", "E")):
        out["p1"][f"{x}-{y}"] = clustered(arms[x][0], arms[y][0], cells)

    # advantage arithmetic (Reviewer 1 item 3)
    r = np.array([1] + [0] * 7, float)
    out["advantage_one_success_in_8"] = float((1 - r.mean()) / (r.std(ddof=1) + 1e-4))

    for st in sorted(glob.glob(str(ROOT / "runs/*/checkpoint-*/trainer_state.json"))):
        run = st.split("/")[-3]
        step = int(st.split("checkpoint-")[1].split("/")[0])
        prev = out["train_logs"].get(run)
        if prev and prev["_step"] >= step:
            continue
        h = json.load(open(st))["log_history"]
        g = lambda k: [e[k] for e in h if k in e]
        clip, ent, zs = g("clip_ratio/region_mean"), g("entropy"), g("frac_reward_zero_std")
        ismin, lpmax, ismean = g("sampling/importance_sampling_ratio/min"), g("sampling/sampling_logp_difference/max"), g("sampling/importance_sampling_ratio/mean")
        out["train_logs"][run] = {
            "_step": step, "n_logged": len(h),
            "clip_region_max": max(clip) if clip else None,
            "entropy_mean": float(np.mean(ent)) if ent else None,
            "frac_zero_std_mean": float(np.mean(zs)) if zs else None,
            "steps_with_masked_sequence": int(sum(v == 0.0 for v in ismin)) if ismin else None,
            "is_weight_mean": float(np.mean(ismean)) if ismean else None,
            "steps_logp_gap_gt10": int(sum(v > 10 for v in lpmax)) if lpmax else None,
            "logp_gap_max_mean": float(np.mean(lpmax)) if lpmax else None}
    dst = ROOT / "results/audit/redteam_recompute.json"
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(out, indent=1, default=str))
    for k in ("s3", "p1", "fisher"):
        for kk, v in out[k].items():
            print(k, kk, {a: (round(b, 3) if isinstance(b, float) else b) for a, b in v.items() if not isinstance(b, list) or len(b) <= 2})
    print("adv", out["advantage_one_success_in_8"], "runs scanned", len(out["train_logs"]))


if __name__ == "__main__":
    main()
