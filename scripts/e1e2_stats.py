"""E1/E2 statistics (PREREGISTRATION_E1E2). Written and frozen before any E1/E2 output existed.

E1. Per model and task: sampled VSC = mean over the 8 sampled episodes (greedy reported separately). The task set is
e1_pp_full + e1_pp_rec (120 tasks, each its own world). Arms: CONTROL s3_matched, ACK s3_ackonly, RSC s3_rv_mix,
RV a12_rv (seeds 0-4).
  P1 RSC - CONTROL, P2 RSC - ACK, P3 ACK - CONTROL: one-sided unpaired seed permutation (252 relabellings), Holm over
  P1-P3; supported if Holm p < 0.05 and the seed-averaged 95% task-bootstrap interval excludes 0
  (scripts/s4_stats.contrast, unpaired).
  T  threshold reading (descriptive, rule fixed here): among the 20 trained runs, split by greedy partial-class VSC on
     s3_conf_full (>= 0.40 = greedy learner, from results/s3/conf). "threshold" if max(sampled VSC of greedy
     non-learners) >= min(sampled VSC of greedy learners); else "separated". Spearman rho also reported.
E2. Per run and condition x workflow: sampled correct rate of the first decisive action (16 samples per context).
  H1 position: per trained run, d = rate(S1V0, receive_pay) - mean(rate(S1V0, order_prepay), rate(S1V0, pay_post)).
     Supported if d < 0 in >= 16 of 20 trained runs (sign test one-sided p = 0.006) and the mean of d over runs has a
     95% bootstrap interval (over runs) below 0.
  H2 computation: per trained run, g = rate(S1V1, receive_pay) - rate(S1V0, receive_pay). "computation helps" if the
     mean of g >= 0.10 with a 95% interval (over runs) above 0; else "not computation".
  H3 chain (descriptive): rate(S2V0, workflow) per arm, and the H1 statistic computed on S2V0.
    python scripts/e1e2_stats.py
"""
from __future__ import annotations

import importlib.util
import json
import random
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("s4_stats", ROOT / "scripts/s4_stats.py")
S4 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(S4)
ARMS = {"CONTROL": "s3_matched", "ACK": "s3_ackonly", "RSC": "s3_rv_mix", "RV": "a12_rv"}
TRAINED = [f"{p}_s{s}" for p in ARMS.values() for s in range(5)]
E1 = {"P1": ("RSC", "CONTROL"), "P2": ("RSC", "ACK"), "P3": ("ACK", "CONTROL")}
SPLITS = ("e1_pp_full", "e1_pp_rec")
PAYFIRST = ("order_prepay", "pay_post")


def e1_load(tag: str) -> tuple[dict | None, dict | None]:
    samp, greedy = defaultdict(list), {}
    for sp in SPLITS:
        p = ROOT / f"results/e1/{tag}__{sp}.jsonl"
        if not p.exists():
            return None, None
        for r in map(json.loads, p.open()):
            v = float(r["score"]["vsc"])
            if r["sample"] < 0:
                greedy[r["task_id"]] = {"vsc": v}
            else:
                samp[r["task_id"]].append(v)
    return {t: {"vsc": sum(v) / len(v)} for t, v in samp.items()}, greedy


def greedy_partial(tag: str) -> float | None:
    m = S4.load(tag, "s3_conf_full", "results/s3/conf")
    if m is None:
        return None
    vals = [v["vsc"] for v in m.values() if v["cls"] in ("partial", "s3:partial:ack", "s3:partial:read")]
    return sum(vals) / len(vals)


def e2_rates(tag: str) -> dict | None:
    p = ROOT / f"results/e2/{tag}.jsonl"
    if not p.exists():
        return None
    acc = defaultdict(lambda: [0, 0])
    for r in map(json.loads, p.open()):
        if r["sample"] >= 0:
            acc[(r["cond"], r["workflow"])][0] += r["correct"]
            acc[(r["cond"], r["workflow"])][1] += 1
    return {k: a / b for k, (a, b) in acc.items()}


def boot_mean(xs: list[float], rng: random.Random, n: int = 10000) -> list[float]:
    ms = sorted(sum(rng.choice(xs) for _ in xs) / len(xs) for _ in range(n))
    return [ms[int(0.025 * n)], ms[int(0.975 * n) - 1]]


def rankdata(x: list[float]) -> list[float]:
    order = sorted(range(len(x)), key=lambda i: x[i])
    r = [0.0] * len(x)
    i = 0
    while i < len(x):
        j = i
        while j + 1 < len(x) and x[order[j + 1]] == x[order[i]]:
            j += 1
        for k in range(i, j + 1):
            r[order[k]] = (i + j) / 2
        i = j + 1
    return r


def spearman(a: list[float], b: list[float]) -> float:
    ra, rb = rankdata(a), rankdata(b)
    ma, mb = sum(ra) / len(ra), sum(rb) / len(rb)
    num = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    den = (sum((x - ma) ** 2 for x in ra) * sum((y - mb) ** 2 for y in rb)) ** 0.5
    return num / den if den else float("nan")


def main() -> None:
    rng = random.Random(0)
    res: dict = {"e1": {}, "e2": {}}
    loaded = {n: [e1_load(f"{p}_s{s}") for s in range(5)] for n, p in ARMS.items()}
    samp = {n: [x[0] for x in v] for n, v in loaded.items()}
    res["e1"]["missing"] = [f"{ARMS[n]}_s{s}" for n, v in samp.items() for s, m in enumerate(v) if m is None]
    res["e1"]["sampled_per_seed"] = {n: [None if m is None else S4.mean(m, "vsc") for m in v] for n, v in samp.items()}
    res["e1"]["greedy_per_seed"] = {n: [None if x[1] is None else S4.mean(x[1], "vsc") for x in v] for n, v in loaded.items()}
    ws, _ = e1_load("warm_start")
    res["e1"]["warm_start_sampled"] = None if ws is None else S4.mean(ws, "vsc")
    ps = {}
    for c, (a, b) in E1.items():
        if all(samp[a]) and all(samp[b]):
            r = S4.contrast(samp[a], samp[b], "vsc", rng, paired=False)
            res["e1"][c] = r
            ps[c] = r["unpaired_perm_p_one_sided"]
    for c, p in (S4.holm(ps) if ps else {}).items():
        res["e1"][c].update(p_holm=p, supported=p < 0.05 and res["e1"][c]["seed_avg"]["ci95"][0] > 0)
    gp = {t: greedy_partial(t) for t in TRAINED}
    sv = {f"{ARMS[n]}_s{s}": m for n, v in res["e1"]["sampled_per_seed"].items() for s, m in enumerate(v)}
    pairs = [(gp[t], sv[t]) for t in TRAINED if gp[t] is not None and sv[t] is not None]
    if pairs:
        learn = [s for g, s in pairs if g >= 0.40]
        non = [s for g, s in pairs if g < 0.40]
        res["e1"]["T"] = {"n_runs": len(pairs), "greedy_learners": len(learn),
                          "spearman_greedy_vs_sampled": spearman([g for g, _ in pairs], [s for _, s in pairs]),
                          "min_sampled_learner": min(learn) if learn else None,
                          "max_sampled_nonlearner": max(non) if non else None,
                          "per_run": {t: {"greedy_partial_s3_conf_full": gp[t], "e1_sampled": sv[t]} for t in TRAINED},
                          "reading": None if not (learn and non) else
                          ("threshold" if max(non) >= min(learn) else "separated")}
    rates = {t: e2_rates(t) for t in TRAINED + ["warm_start"]}
    res["e2"]["missing"] = [t for t, r in rates.items() if r is None]
    have = [t for t in TRAINED if rates[t]]
    res["e2"]["per_run"] = {t: {f"{c}|{w}": v for (c, w), v in sorted(rates[t].items())} for t in rates if rates[t]}

    def pos(t: str, cond: str) -> float:
        r = rates[t]
        return r[(cond, "receive_pay")] - sum(r[(cond, w)] for w in PAYFIRST) / 2
    if have:
        d = [pos(t, "S1V0") for t in have]
        res["e2"]["H1"] = {"n_runs": len(d), "n_negative": sum(x < 0 for x in d), "mean": sum(d) / len(d),
                           "ci95_over_runs": boot_mean(d, rng)}
        res["e2"]["H1"]["supported"] = res["e2"]["H1"]["n_negative"] >= 16 and res["e2"]["H1"]["ci95_over_runs"][1] < 0
        g = [rates[t][("S1V1", "receive_pay")] - rates[t][("S1V0", "receive_pay")] for t in have]
        ci = boot_mean(g, rng)
        res["e2"]["H2"] = {"n_runs": len(g), "mean": sum(g) / len(g), "ci95_over_runs": ci,
                           "reading": "computation helps" if sum(g) / len(g) >= 0.10 and ci[0] > 0 else "not computation"}
        d2 = [pos(t, "S2V0") for t in have]
        res["e2"]["H3"] = {"S2V0_position_mean": sum(d2) / len(d2), "S2V0_ci95_over_runs": boot_mean(d2, rng),
                           "S2V0_by_arm_workflow": {n: {w: sum(rates[f"{p}_s{s}"][("S2V0", w)] for s in range(5)
                                                               if rates[f"{p}_s{s}"]) / max(1, sum(bool(rates[f"{p}_s{s}"]) for s in range(5)))
                                                        for w in ("order_prepay", "pay_post", "receive_pay")}
                                                    for n, p in ARMS.items()}}
    (ROOT / "results/e1e2_summary.json").write_text(json.dumps(res, indent=2))
    for c in E1:
        r = res["e1"].get(c)
        if r:
            print("E1", c, f"{r['seed_avg']['est']:+.3f}", [round(x, 3) for x in r["seed_avg"]["ci95"]],
                  "p", round(r["unpaired_perm_p_one_sided"], 4), "holm", round(r["p_holm"], 4), r["supported"])
    print("E1 T", (res["e1"].get("T") or {}).get("reading"), "| E2 H1", (res["e2"].get("H1") or {}).get("supported"),
          "| E2 H2", (res["e2"].get("H2") or {}).get("reading"), "| missing", res["e1"]["missing"], res["e2"]["missing"])


if __name__ == "__main__":
    main()
