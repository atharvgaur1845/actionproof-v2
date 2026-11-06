"""Confirmatory analysis exactly as pre-registered (PREREGISTRATION.md §2). Run ONCE on P after G3.

    python -m actionproof.eval.confirmatory --dir results/eval/conf --out results/confirmatory.json
    # dry run on the (already inspected) development sets:
    python -m actionproof.eval.confirmatory --dir results/eval/r3 --splits test_iid,test_ood_fault --out /tmp/dry.json

Uses the registered `stats.hierarchical_bootstrap` unchanged for H1 (baseline averaged over its seeds).
A sensitivity version that also resamples the baseline's seeds is reported alongside, and is not the test.
Holm correction over the key-secondary family {H2, H7, H8}.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from actionproof.eval.calibration import TRAIN_RHO, profile, spearman
from actionproof.eval.stats import hierarchical_bootstrap, paired_bootstrap

ITERS = 10_000
ARMS = {  # arm -> run-name template per seed; seeds actually present are used
    "rft_v2": "rft_tiny_v2",
    "pcstr_d": "grpo_tiny_v2_pcstr_d_s{s}",
    "outcome": "grpo_tiny_v2_outcome_s{s}",
    "tool_trust": "grpo_tiny_v2_tool_trust_s{s}",
    "pcstr": "grpo_tiny_v2_pcstr_s{s}",
    "pcstr_c": "grpo_tiny_v2_pcstr_c_s{s}",
    "llm_judge": "grpo_tiny_v2_llm_judge_s{s}",
}
BASELINE = "outcome"  # selected on val before any look at P (SPEC §27, 160_baseline_select_val)


def load(d: Path, run: str, splits: list[str]) -> dict[str, dict] | None:
    out = {}
    for sp in splits:
        f = d / f"{run}__neutral__{sp}.jsonl"
        if not f.exists():
            return None
        for line in open(f):
            r = json.loads(line)
            out[r["task_id"]] = r
    return out


def seeds_of(d: Path, arm: str, splits: list[str]) -> list[dict[str, dict]]:
    t = ARMS[arm]
    if "{s}" not in t:
        r = load(d, t, splits)
        return [r] if r else []
    return [r for s in range(5) if (r := load(d, t.format(s=s), splits))]


def vsc(eps: dict[str, dict], keep=lambda r: True) -> dict[str, float]:
    return {k: float(r["vsc"]) for k, r in eps.items() if keep(r)}


def boot_p(deltas: list[float]) -> float:
    """Two-sided bootstrap p-value: twice the share of resampled deltas on the far side of 0."""
    lo = sum(x <= 0 for x in deltas) / len(deltas)
    hi = sum(x >= 0 for x in deltas) / len(deltas)
    return min(1.0, 2 * min(lo, hi))


def hier_deltas(a_seeds, b_seeds, metric, ids, seed=0) -> list[float]:
    """Resample seeds of BOTH arms, then tasks; `metric(seed_eps, task_ids) -> float`."""
    rng = random.Random(seed)
    out = []
    for _ in range(ITERS):
        t = [ids[rng.randrange(len(ids))] for _ in ids]
        a = [a_seeds[rng.randrange(len(a_seeds))] for _ in a_seeds]
        b = [b_seeds[rng.randrange(len(b_seeds))] for _ in b_seeds]
        out.append(sum(metric(s, t) for s in a) / len(a) - sum(metric(s, t) for s in b) / len(b))
    return out


def mean_vsc(eps, t):
    return sum(float(eps[i]["vsc"]) for i in t) / len(t)


def fcr(eps, t):
    """P(reported completed | G=0) on this model's own G=0 tasks among t."""
    g0 = [i for i in t if eps[i]["verdict"]["G"] == 0]
    return sum(eps[i]["verdict"]["reported"] == "completed" for i in g0) / len(g0) if g0 else 0.0


def ci(xs):
    xs = sorted(xs)
    return [round(xs[int(0.025 * len(xs))], 4), round(xs[int(0.975 * len(xs)) - 1], 4)]


def holm(ps: dict[str, float], alpha=0.05) -> dict[str, bool]:
    order = sorted(ps, key=ps.get)
    out, stop = {}, False
    for i, h in enumerate(order):
        stop = stop or ps[h] > alpha / (len(order) - i)
        out[h] = not stop
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="results/eval/conf")
    ap.add_argument("--splits", default="test_iid_c,test_ood_fault_c")
    ap.add_argument("--out", default="results/confirmatory.json")
    a = ap.parse_args()
    d, splits = Path(a.dir), a.splits.split(",")
    S = {arm: seeds_of(d, arm, splits) for arm in ARMS}
    print("seeds found:", {k: len(v) for k, v in S.items()})
    res: dict = {"splits": splits, "seeds": {k: len(v) for k, v in S.items()}}
    ids = sorted(set.intersection(*[set(s) for v in S.values() for s in v]))
    res["n_tasks"] = len(ids)

    # H1 (single primary test): VSC(pcstr_d) > VSC(baseline) on P.
    A = [vsc(s) for s in S["pcstr_d"]]
    B = [vsc(s) for s in S[BASELINE]]
    bmean = {i: sum(b[i] for b in B) / len(B) for i in ids}
    pt, lo, hi = hierarchical_bootstrap(A, bmean, iters=ITERS)
    sens = hier_deltas(S["pcstr_d"], S[BASELINE], mean_vsc, ids)
    res["H1"] = {"baseline": BASELINE, "delta": round(pt, 4), "ci95": [round(lo, 4), round(hi, 4)],
                 "claim": lo > 0 or hi < 0, "sensitivity_both_seeds_ci95": ci(sens), "p": boot_p(sens),
                 "means": {"pcstr_d": round(sum(map(mean_vsc, S["pcstr_d"], [ids] * len(A))) / len(A), 4),
                           BASELINE: round(sum(bmean.values()) / len(ids), 4)}}

    # H2: GRPO+tool_trust raises FCR vs its init (each model conditioned on its own G=0 tasks).
    h2 = hier_deltas(S["tool_trust"], S["rft_v2"], fcr, ids, seed=1)
    pt2 = sum(fcr(s, ids) for s in S["tool_trust"]) / len(S["tool_trust"]) - fcr(S["rft_v2"][0], ids)
    res["H2"] = {"delta_fcr": round(pt2, 4), "ci95": ci(h2), "p": boot_p(h2)}

    # H7: evaluated on probe_cues (interp/h7_compare.py); probe did not beat the text baseline -> not supported.
    res["H7"] = {"p": 1.0, "note": "probe AUROC CI vs text baseline includes 0 (SPEC §27); not supported"}

    # H8: Spearman(check rate, rho_s) CI > 0 for pcstr_c AND VSC non-inferior to pcstr_d by -2 pp.
    if S["pcstr_c"]:
        eps = [S["pcstr_c"][0][i] for i in ids]
        prof = profile(eps)
        svcs = sorted(s for s in prof if prof[s][1] > 0)
        r = spearman([prof[s][0] / prof[s][1] for s in svcs], [TRAIN_RHO[s] for s in svcs])
        rng, rs = random.Random(2), []
        for _ in range(2000):
            p = profile([eps[rng.randrange(len(eps))] for _ in eps])
            ss = [s for s in svcs if p.get(s, (0, 0))[1] > 0]
            v = spearman([p[s][0] / p[s][1] for s in ss], [TRAIN_RHO[s] for s in ss])
            if v is not None:
                rs.append(v)
        ni = hier_deltas(S["pcstr_c"], S["pcstr_d"], mean_vsc, ids, seed=3)
        p_rho = min(1.0, 2 * sum(x <= 0 for x in rs) / len(rs))
        p_ni = min(1.0, 2 * sum(x <= -0.02 for x in ni) / len(ni))
        res["H8"] = {"spearman": r, "spearman_ci95": ci(rs), "noninf_delta_ci95": ci(ni), "margin": -0.02,
                     "p": max(p_rho, p_ni), "met": ci(rs)[0] > 0 and ci(ni)[0] > -0.02, "n_services": len(svcs),
                     "caveat": "with 3 services Spearman takes only a few values; test_calib_swap shows the profile "
                               "is a memorised service prior (expert demos encode it), so a pass is not evidence of "
                               "in-context calibration"}
    else:
        res["H8"] = {"p": 1.0, "note": "pcstr_c not evaluated"}
    res["holm_reject"] = holm({h: res[h]["p"] for h in ("H2", "H7", "H8")})

    # Exploratory, no multiplicity claim.
    benign = [i for i in ids if S["pcstr_d"][0][i]["benign"]]
    res["H3_benign"] = dict(zip(("delta", "lo", "hi"), hierarchical_bootstrap(
        [{i: s[i] for i in benign} for s in A], {i: bmean[i] for i in benign}, iters=ITERS)))
    res["H3_benign"]["noninferior_5pp"] = res["H3_benign"]["lo"] > -0.05
    ood = [i for i in ids if "ood_fault" in S["pcstr_d"][0][i]["split"]]
    if ood:
        res["H4_ood_fault"] = dict(zip(("delta", "lo", "hi"), hierarchical_bootstrap(
            [{i: s[i] for i in ood} for s in A], {i: bmean[i] for i in ood}, iters=ITERS)))
    tt = [vsc(s) for s in S["tool_trust"]]
    res["pcstr_d_vs_tool_trust"] = dict(zip(("delta", "lo", "hi"), hierarchical_bootstrap(
        A, {i: sum(t[i] for t in tt) / len(tt) for i in ids}, iters=ITERS)))
    res["per_arm_vsc"] = {arm: [round(mean_vsc(s, ids), 4) for s in v] for arm, v in S.items() if v}
    res["pairwise_seed0_vs_baseline"] = {arm: [round(x, 4) for x in paired_bootstrap(vsc(v[0]), B[0])]
                                         for arm, v in S.items() if v and arm != BASELINE}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=1))
    print(json.dumps({k: res[k] for k in ("n_tasks", "H1", "H2", "H8", "holm_reject", "per_arm_vsc")}, indent=1))


if __name__ == "__main__":
    main()
