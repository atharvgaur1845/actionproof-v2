"""Analysis of the registered testbed families, exactly as registered (V2TB_PROTOCOL.md §4-5).

    python scripts/ad2/tab_analyze.py TB1 | TB2 | TB3 | TB4 | TB6

Reads results/ad2/tabular/<FAM>.jsonl, writes results/ad2/tabular/<FAM>_summary.json.
"""
from __future__ import annotations

import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev

ROOT = Path(__file__).resolve().parents[2]
D = ROOT / "results/ad2/tabular"
VOI_MARGIN = 0.02
SFB_POOL = ("nostd", "rloo", "critic_v", "critic_q", "ema", "ctxema", "lagr")  # non-privileged baselines
CANDS = ("cvc", "fork_bel", "fork_act", "igc", "ctxema", "awrbn")


def load(fam: str) -> list[dict]:
    return [json.loads(l) for l in (D / f"{fam}.jsonl").read_text().splitlines() if l.strip()]


def agree(vp: float, voi: float) -> bool | None:
    if abs(voi) <= VOI_MARGIN:
        return None
    return (vp > 0.5) == (voi > 0)


def tb1(rows: list[dict]) -> dict:
    by = defaultdict(list)  # est -> list of (p, c, s, L, seed, vp_risky, vp_safe, voi, regret)
    for r in rows:
        _, p, c, s, L, name, seed = r["id"].split("|")
        e = next(iter(r["eval"].values()))
        by[name].append((float(p), float(c), float(s), float(L), int(seed), e["verify_prob"][0], e["verify_prob"][1],
                         e["voi"][0], e["regret"]))
    out = {"per_est": {}}
    for name, xs in by.items():
        pos = [x for x in xs if x[1] >= 0.01]
        ag = [agree(x[5], x[7]) for x in pos]
        ag = [a for a in ag if a is not None]
        # c-insensitivity: per (p, s, L) cell, range over c >= 0.01 of the seed-mean verify prob
        cell = defaultdict(lambda: defaultdict(list))
        for x in pos:
            cell[(x[0], x[2], x[3])][x[1]].append(x[5])
        rng = [max(mean(v) for v in d.values()) - min(mean(v) for v in d.values()) for d in cell.values()]
        out["per_est"][name] = {
            "n": len(xs), "agreement_c_pos": sum(ag) / len(ag) if ag else None,
            "wrong_rate_c_pos": 1 - sum(ag) / len(ag) if ag else None,
            "mean_regret_c_pos": mean(x[8] for x in pos),
            "frac_cells_c_insensitive": sum(r_ < 0.25 for r_ in rng) / len(rng) if rng else None,
            "safe_verify_mean": mean(x[6] for x in xs),
            "c0_verify_frac_ps_pos": (lambda z: sum(x[5] > 0.5 for x in z) / len(z) if z else None)(
                [x for x in xs if x[1] == 0 and x[0] * x[2] >= 0.05]),
            "never_verify_frac_c_pos": sum(x[5] < 0.1 for x in pos) / len(pos),
        }
        mirror = [x for x in pos if x[0] < 0.5 and x[0] * (1 + x[3]) > x[1] + VOI_MARGIN and x[7] < -VOI_MARGIN]
        out["per_est"][name]["mirror_cells_verify_frac"] = (sum(x[5] > 0.5 for x in mirror) / len(mirror)) if mirror else None
        out["per_est"][name]["n_mirror"] = len(mirror)
    P = out["per_est"]
    g = lambda n, k: (P.get(n) or {}).get(k)
    H = {
        "H1a_grpo_c_insensitive": (g("grpo", "frac_cells_c_insensitive") or 0) >= 0.8,
        "H1b_grpo_wrong_ge_0.30": (g("grpo", "wrong_rate_c_pos") or 0) >= 0.30,
        "H1b_nostd_wrong_le_0.15": (g("nostd", "wrong_rate_c_pos") or 1) <= 0.15,
        "H1c_grpoB_wrong_le_0.15": (g("grpo_B", "wrong_rate_c_pos") or 1) <= 0.15,
        "H1d_grpo_c0_verifies_ge_0.9": (g("grpo", "c0_verify_frac_ps_pos") or 0) >= 0.9,
        "H1e_gdpo_never_verifies_ge_0.9": (g("gdpo", "never_verify_frac_c_pos") or 0) >= 0.9,
        "H1f_batchstd_wrong_gt_nostd": (g("batchstd", "wrong_rate_c_pos") or 0) > (g("nostd", "wrong_rate_c_pos") or 1),
        "H1g_awrb_mirror_ge_0.5": (g("awrb", "mirror_cells_verify_frac") or 0) >= 0.5,
        "H1g_awrbn_mirror_le_0.2": (g("awrbn", "mirror_cells_verify_frac") or 1) <= 0.2,
    }
    out["hypotheses"] = H
    return out


def _seed_metrics(rows: list[dict], key_fn) -> dict:
    """key -> seed -> list of per-task eval dicts."""
    acc = defaultdict(lambda: defaultdict(list))
    eps = defaultdict(list)
    for r in rows:
        k, seed = key_fn(r)
        for tag, e in r["eval"].items():
            acc[k][seed].append(e)
        eps[k].append(r["episodes_used"])
    return acc, {k: mean(v) for k, v in eps.items()}


def _agg(evs: list[dict], field: str) -> float | None:
    xs = [e[field] for e in evs if e.get(field) is not None]
    return mean(xs) if xs else None


def tb2(rows: list[dict]) -> dict:
    def key(r):
        _, mode, c, L, name, var, seed = r["id"].split("|")
        return (name, var), int(seed)
    acc, eps = _seed_metrics(rows, key)
    fields = ("regret", "J", "safe_verify_rate", "risky_verify_rate", "mc_dupl", "precision", "recall",
              "uninformative_read_frac", "recovery_given_silent", "mc_cost", "cost_per_completed", "mc_dup_any")
    tab = {}
    for k, seeds in acc.items():
        per_seed = {s: {f: _agg(v, f) for f in fields} for s, v in seeds.items()}
        tab["/".join(k)] = {"episodes": eps[k], "seeds": len(per_seed),
                            **{f: mean(x[f] for x in per_seed.values() if x[f] is not None) for f in fields},
                            "regret_by_seed": {s: x["regret"] for s, x in sorted(per_seed.items())}}
    gates = {}
    for cand in CANDS:
        variants = [k for k in tab if k.split("/")[0] == cand]
        for cv in variants:
            ce = tab[cv]["episodes"]
            best = None
            for b in SFB_POOL:
                opts = sorted([k for k in tab if k.split("/")[0] == b and k.split("/")[1].startswith("x")],
                              key=lambda k: tab[k]["episodes"])
                if not opts:
                    continue
                matched = next((k for k in opts if tab[k]["episodes"] >= ce), opts[-1])
                if cv.startswith(b + "/"):
                    matched = cv  # a baseline-pool member compared with the others at its own budget
                    continue
                if best is None or tab[matched]["regret"] < tab[best]["regret"]:
                    best = matched
            if best is None:
                continue
            c_rs, b_rs = tab[cv]["regret_by_seed"], tab[best]["regret_by_seed"]
            common = sorted(set(c_rs) & set(b_rs))
            diffs = [b_rs[s] - c_rs[s] for s in common]
            sfb_reg = mean(b_rs[s] for s in common)
            G1 = bool(common) and mean(diffs) >= 0.2 * sfb_reg and sum(d > 0 for d in diffs) >= 4
            G2 = (tab[cv]["safe_verify_rate"] or 0) <= (tab[best]["safe_verify_rate"] or 0) + 0.02
            G3 = (tab[cv]["mc_dupl"] or 0) <= (tab[best]["mc_dupl"] or 0) + 0.01
            gates[cv] = {"sfb": best, "sfb_episodes": tab[best]["episodes"], "cand_episodes": ce,
                         "budget_matched": tab[best]["episodes"] >= ce, "regret_cand": tab[cv]["regret"],
                         "regret_sfb": tab[best]["regret"], "seed_diffs": diffs, "G1": G1, "G2": G2, "G3": G3,
                         "pass_T1_T3": G1 and G2 and G3}
    return {"table": tab, "gates": gates}


def tb3(rows: list[dict]) -> dict:
    out = defaultdict(lambda: defaultdict(list))
    eps = defaultdict(list)
    for r in rows:
        _, name, var, seed = r["id"].split("|")
        k = f"{name}/{var}"
        eps[k].append(r["episodes_used"])
        for tag, e in r["eval"].items():
            split = "train" if tag.startswith("gen_train") else "_".join(tag.split("_")[1:3])
            ag = [agree(vp, v) for vp, v in zip(e["verify_prob"], e["voi"])]
            ag = [a for a in ag if a is not None]
            out[k][split].append({"regret": e["regret"], "agree": (sum(ag) / len(ag)) if ag else None,
                                  "safe": e["verify_prob"][-1]})
    res = {}
    for k, sp in out.items():
        res[k] = {"episodes": mean(eps[k])}
        for s, xs in sp.items():
            res[k][s] = {"regret": mean(x["regret"] for x in xs),
                         "agree": mean(x["agree"] for x in xs if x["agree"] is not None),
                         "safe_verify": mean(x["safe"] for x in xs)}
    return {"table": res}


def tb4(rows: list[dict]) -> dict:
    acc = defaultdict(list)
    for r in rows:
        _, name, prior, seed = r["id"].split("|")
        evs = list(r["eval"].values())
        acc[(name, prior)].append({"regret": mean(e["regret"] for e in evs),
                                   "safe": mean(e["safe_verify_rate"] or 0 for e in evs),
                                   "risky": mean(e["risky_verify_rate"] or 0 for e in evs)})
    return {"table": {f"{n}/{p}": {"regret": mean(x["regret"] for x in v), "regret_sd": pstdev([x["regret"] for x in v]),
                                   "safe_verify": mean(x["safe"] for x in v), "risky_verify": mean(x["risky"] for x in v),
                                   "n": len(v)} for (n, p), v in sorted(acc.items())}}


def tb6(rows: list[dict]) -> dict:
    acc = defaultdict(list)
    for r in rows:
        _, p, c, est, rho0, G, opt, cond, seed = r["id"].split("|")
        e = next(iter(r["eval"].values()))
        a = agree(e["verify_prob"][0], e["voi"][0])
        acc[(est, cond, rho0, G, opt)].append((a, e["regret"], e["verify_prob"][0]))
    out = {}
    for k, v in sorted(acc.items()):
        ag = [x[0] for x in v if x[0] is not None]
        out["/".join(k)] = {"agreement": sum(ag) / len(ag) if ag else None, "regret": mean(x[1] for x in v),
                            "verify": mean(x[2] for x in v), "n": len(v)}
    return {"table": out}


def main() -> None:
    fam = sys.argv[1]
    rows = load(fam)
    res = {"TB1": tb1, "TB2": tb2, "TB3": tb3, "TB4": tb4, "TB6": tb6}[fam](rows)
    res["n_runs"] = len(rows)
    (D / f"{fam}_summary.json").write_text(json.dumps(res, indent=1, default=float))
    print(json.dumps(res.get("hypotheses", {}), indent=1))
    if fam == "TB2":
        print(json.dumps(res["gates"], indent=1))
    print(f"{fam}: {len(rows)} runs summarised")


if __name__ == "__main__":
    main()
