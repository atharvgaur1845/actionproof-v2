"""RG1 registered analysis (PREREGISTRATION_RG1.md). Written and hashed before any sealed-split RG1 output exists.

Episode files: results/rg1/<tag>__<split>.jsonl (scripts/rg1_eval.py: 1 greedy + 2 sampled episodes per task;
scripts/rg1_policies_eval.py: 1 episode per task). Every episode carries `txn` flags (actionproof.txn.truth).

Arms on the sealed split rg1_test (440 tasks): RG_D_s{k} = D seed k behind the full gate; NG_D_s{k} = D seed k with no
gate, same loop and budgets; k = 0, 1, 2.

Unit and statistic (as SEL1). A model's task score is the mean of the flag over that task's episodes; an arm's task
score is the mean over its three seed models; differences are paired on tasks. Bootstrap: B = 10,000 cluster resamples
(numpy default_rng, seed 20261128); clusters are `meta.txn.quad_id`, else `template_id`; statistic = task-weighted
mean of the per-task difference; 95% percentile interval; one-sided p = (1 + #{b: boot <= 0}) / (B + 1).

Confirmatory family (Holm over H1-H3; each supported iff Holm-adjusted one-sided p < 0.05 AND all three same-seed
differences have the registered sign):
- H1 false completion: FCR over all 440 tasks, NG - RG > 0.
- H2 escalation where the record cannot be read: escalation rate on the 60 unresolvable tasks, RG - NG > 0.
- H3 safe resolution: STR over all 440 tasks, RG - NG > 0.
- H4 (non-inferiority, outside the family): VTC on the 380 resolvable tasks (regime standard or delayed); supported
  iff the lower 95% bound of RG - NG is above -0.02.
RG1 is supported overall iff H1, H2, H3 and H4 all hold.

Registered secondary comparisons (estimates and 95% intervals, no test, no multiplicity claim):
- S1 RG vs the scripted reference B4 (one episode per task) on rg1_test: VTC (resolvable), STR, unsafe, FCR; overall
  and by group (B4 runs on canonical tool names even in the renamed-surface groups).
- S2 rule ablations, D seed 0 (AB_<rule>_D_s0 vs RG_D_s0): FCR, DMR, unsafe, VTC, STR, escalation (unresolvable).
- S3 other agents with and without the gate: base model, Dw_aug seed 0, SEL1's SEL seed 0, Aya Expanse 8B
  (untrained), scripted B0, B1, B3.
- S4 txn_test replication: RG_D_s{k}__txn_test vs TXN1's D_s{k} episodes (results/txn); txn_test outcomes for D
  were known before RG1, so this is descriptive.
Descriptive tables for every tag present: TXN metrics with numerators and denominators (truth.summarise), by group
and regime; cost (calls, reads, writes, gate calls); gate events.

    python scripts/rg1_stats.py          # writes results/rg1/rg1_summary.json and .md
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
R = ROOT / "results/rg1"
B, SEED, SEEDS = 10_000, 20261128, (0, 1, 2)
RULES = ("no_verify", "no_retry", "no_fresh", "no_repair", "no_escalate")
FLAGS = ("vtc", "str", "fcr", "dmr", "unsafe", "escalated", "urr", "fcompr")


def load(tag: str, split: str, res: Path = R) -> dict[str, list[dict]] | None:
    f = res / f"{tag}__{split}.jsonl"
    if not f.exists():
        return None
    out: dict[str, list[dict]] = defaultdict(list)
    for line in f.read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            out[r["task_id"]].append(r["txn"])
    return dict(out)


def task_meta(split: str) -> dict[str, dict]:
    from actionproof.bench.generator_proc import DATA
    out = {}
    for line in (DATA / f"{split}.jsonl").read_text().splitlines():
        if line.strip():
            d = json.loads(line)
            out[d["task_id"]] = d["meta"]["txn"]
    return out


def cluster_of(tx: dict) -> str:
    return tx.get("quad_id") or tx["template_id"]


def rate(eps: list[dict], flag: str) -> float:
    return float(np.mean([bool(e[flag]) for e in eps]))


class Boot:
    def __init__(self, tasks: list[str], meta: dict[str, dict], rng: np.random.Generator) -> None:
        cl = sorted({cluster_of(meta[t]) for t in tasks})
        idx = {c: i for i, c in enumerate(cl)}
        self.member = np.array([idx[cluster_of(meta[t])] for t in tasks])
        draws = rng.integers(0, len(cl), size=(B, len(cl)))
        self.w = np.zeros((B, len(cl)))
        for j in range(len(cl)):
            self.w[:, j] = (draws == j).sum(axis=1)
        self.n_clusters = len(cl)

    def means(self, x: np.ndarray) -> np.ndarray:
        sums = np.bincount(self.member, weights=x, minlength=self.n_clusters)
        cnts = np.bincount(self.member, minlength=self.n_clusters).astype(float)
        return (self.w @ sums) / (self.w @ cnts)


def scores(data: dict[str, list[dict]], tasks: list[str], flag: str) -> np.ndarray:
    return np.array([rate(data[t], flag) for t in tasks])


def p_one_sided(boot: np.ndarray) -> float:
    return float((1 + np.sum(boot <= 0)) / (B + 1))


def holm(ps: dict[str, float]) -> dict[str, float]:
    order = sorted(ps, key=ps.get)
    out, run = {}, 0.0
    for i, h in enumerate(order):
        run = max(run, min(1.0, (len(order) - i) * ps[h]))
        out[h] = run
    return out


def compare(a: list[dict], b: list[dict], tasks: list[str], flag: str, boot: Boot) -> dict:
    """Paired difference mean(a) - mean(b); a and b are lists of per-model episode maps (seeds)."""
    sa = [scores(x, tasks, flag) for x in a]
    sb = [scores(x, tasks, flag) for x in b]
    ma, mb = np.mean(sa, axis=0), np.mean(sb, axis=0)
    d = ma - mb
    bs = boot.means(d)
    out = {"flag": flag, "n_tasks": len(tasks), "n_clusters": boot.n_clusters, "a": float(ma.mean()),
           "b": float(mb.mean()), "diff": float(d.mean()), "ci95": [float(np.quantile(bs, .025)),
                                                                   float(np.quantile(bs, .975))],
           "p_one_sided": p_one_sided(bs)}
    if len(sa) == len(sb) > 1:
        out["seed_diffs"] = [float((x - y).mean()) for x, y in zip(sa, sb)]
        out["a_seeds"], out["b_seeds"] = [float(x.mean()) for x in sa], [float(y.mean()) for y in sb]
    return out


def tags_present(split: str) -> list[str]:
    return sorted(p.name.split("__")[0] for p in R.glob(f"*__{split}.jsonl"))


def descriptive(split: str, meta: dict[str, dict]) -> dict:
    from actionproof.txn.truth import breakdown, summarise
    out = {}
    for tag in tags_present(split):
        rows = [json.loads(line) for line in (R / f"{tag}__{split}.jsonl").read_text().splitlines() if line.strip()]
        sc = [r["txn"] for r in rows]
        g = [r.get("gate") or {} for r in rows]
        ev: dict[str, int] = defaultdict(int)
        for x in g:
            for e in x.get("log") or []:
                ev[f"{e['rule']}:{e['action']}"] += 1
        out[tag] = {"n_episodes": len(rows), "all": summarise(sc), "by_group": breakdown(sc, "group"),
                    "by_regime": breakdown(sc, "regime"), "gate_events": dict(sorted(ev.items())),
                    "gate_calls_per_episode": float(np.mean([x.get("gate_calls", 0) for x in g])) if g else 0.0}
    return out


def main() -> None:
    split = "rg1_test"
    meta = task_meta(split)
    rg = [load(f"RG_D_s{k}", split) for k in SEEDS]
    ng = [load(f"NG_D_s{k}", split) for k in SEEDS]
    missing = [f"{p}_D_s{k}" for p, arm in (("RG", rg), ("NG", ng)) for k, d in zip(SEEDS, arm)
               if d is None or len(d) != len(meta)]
    if missing:
        sys.exit(f"missing or incomplete confirmatory episode files: {missing}")
    rng = np.random.default_rng(SEED)
    allt = sorted(meta)
    unres = sorted(t for t in allt if meta[t]["regime"] == "unresolvable")
    resv = sorted(t for t in allt if meta[t]["regime"] in ("standard", "delayed"))
    boots = {"ALL": Boot(allt, meta, rng), "U": Boot(unres, meta, rng), "RES": Boot(resv, meta, rng)}
    res: dict = {"B": B, "seed": SEED, "split": split, "n_tasks": {"ALL": len(allt), "U": len(unres), "RES": len(resv)}}
    res["H1"] = compare(ng, rg, allt, "fcr", boots["ALL"])          # NG - RG > 0
    res["H2"] = compare(rg, ng, unres, "escalated", boots["U"])     # RG - NG > 0
    res["H3"] = compare(rg, ng, allt, "str", boots["ALL"])          # RG - NG > 0
    ph = holm({h: res[h]["p_one_sided"] for h in ("H1", "H2", "H3")})
    for h in ("H1", "H2", "H3"):
        res[h]["p_holm"] = ph[h]
        res[h]["all_seed_diffs_positive"] = all(x > 0 for x in res[h]["seed_diffs"])
        res[h]["supported"] = bool(ph[h] < 0.05 and res[h]["all_seed_diffs_positive"])
    res["H1"]["direction"], res["H2"]["direction"], res["H3"]["direction"] = "NG - RG", "RG - NG", "RG - NG"
    res["H4"] = compare(rg, ng, resv, "vtc", boots["RES"])
    res["H4"].update(direction="RG - NG", bound=-0.02, supported=bool(res["H4"]["ci95"][0] > -0.02))
    res["RG1_supported"] = bool(all(res[h]["supported"] for h in ("H1", "H2", "H3", "H4")))

    # S1: RG vs B4 (scripted reference, one episode per task)
    b4 = load("B4", split)
    if b4 is not None and len(b4) == len(meta):
        res["S1"] = {"vtc_resolvable": compare(rg, [b4], resv, "vtc", boots["RES"]),
                     **{f: compare(rg, [b4], allt, f, boots["ALL"]) for f in ("str", "unsafe", "fcr")},
                     "by_group": {}}
        for grp in sorted({meta[t]["group"] for t in allt}):
            ts = [t for t in allt if meta[t]["group"] == grp]
            bg = Boot(ts, meta, np.random.default_rng(SEED))
            res["S1"]["by_group"][grp] = {f: compare(rg, [b4], ts, f, bg) for f in ("str", "vtc", "unsafe")}
    # S2: rule ablations on D seed 0
    full0 = rg[0]
    res["S2"] = {}
    for rule in RULES:
        ab = load(f"AB_{rule}_D_s0", split)
        if ab is None or len(ab) != len(meta):
            continue
        res["S2"][rule] = {**{f: compare([ab], [full0], allt, f, boots["ALL"]) for f in ("fcr", "dmr", "unsafe", "str")},
                           "vtc_resolvable": compare([ab], [full0], resv, "vtc", boots["RES"]),
                           "escalated_unresolvable": compare([ab], [full0], unres, "escalated", boots["U"])}
    # S3: other agents with and without the gate
    res["S3"] = {}
    for gate_tag, plain_tag in (("RG_base", "NG_base"), ("RG_Dwaug_s0", "NG_Dwaug_s0"), ("RG_SEL_s0", "NG_SEL_s0"),
                                ("RG_aya8b", "NG_aya8b"), ("B0_gate", "B0"),
                                ("B1_gate", "B1"), ("B3_gate", "B3")):
        a, b = load(gate_tag, split), load(plain_tag, split)
        if a is None or b is None or len(a) != len(meta) or len(b) != len(meta):
            continue
        res["S3"][gate_tag] = {**{f: compare([a], [b], allt, f, boots["ALL"]) for f in ("fcr", "unsafe", "str")},
                               "vtc_resolvable": compare([a], [b], resv, "vtc", boots["RES"]),
                               "escalated_unresolvable": compare([a], [b], unres, "escalated", boots["U"])}
    # S4: txn_test replication against TXN1's D episodes
    tmeta = task_meta("txn_test")
    rgt = [load(f"RG_D_s{k}", "txn_test") for k in SEEDS]
    dt = [load(f"D_s{k}", "txn_test", ROOT / "results/txn") for k in SEEDS]
    if all(x is not None and len(x) == len(tmeta) for x in rgt + dt):
        tall = sorted(tmeta)
        tun = sorted(t for t in tall if tmeta[t]["regime"] == "unresolvable")
        tres = sorted(t for t in tall if tmeta[t]["regime"] in ("standard", "delayed"))
        r2 = np.random.default_rng(SEED)
        tb = {"ALL": Boot(tall, tmeta, r2), "U": Boot(tun, tmeta, r2), "RES": Boot(tres, tmeta, r2)}
        res["S4"] = {"fcr": compare(dt, rgt, tall, "fcr", tb["ALL"]), "str": compare(rgt, dt, tall, "str", tb["ALL"]),
                     "escalated_unresolvable": compare(rgt, dt, tun, "escalated", tb["U"]),
                     "vtc_resolvable": compare(rgt, dt, tres, "vtc", tb["RES"]),
                     "unsafe": compare(dt, rgt, tall, "unsafe", tb["ALL"]),
                     "note": "fcr and unsafe are D - RG; the others RG - D"}
    res["descriptive"] = {split: descriptive(split, meta), "txn_test": descriptive("txn_test", tmeta)}
    (R / "rg1_summary.json").write_text(json.dumps(res, indent=1, sort_keys=True) + "\n")
    (R / "rg1_summary.md").write_text(markdown(res))
    print(json.dumps({h: {k: res[h].get(k) for k in ("diff", "ci95", "p_holm", "seed_diffs", "supported")}
                      for h in ("H1", "H2", "H3", "H4")}, indent=1))
    print("RG1_supported:", res["RG1_supported"])


def _f(x: float) -> str:
    return f"{x:.3f}"


def _c(r: dict) -> str:
    return f"{_f(r['diff'])} [{_f(r['ci95'][0])}, {_f(r['ci95'][1])}]"


def markdown(res: dict) -> str:
    L = ["# RG1 registered analysis (scripts/rg1_stats.py)", "",
         f"Split {res['split']}: {res['n_tasks']['ALL']} tasks ({res['n_tasks']['U']} unresolvable, "
         f"{res['n_tasks']['RES']} resolvable). Arms: RG = D behind the full gate, NG = D with no gate; seeds 0-2; "
         "1 greedy + 2 sampled episodes per task. Cluster bootstrap, B = 10,000.", "",
         "| Hypothesis | Direction | Arm A | Arm B | Difference [95% CI] | Holm p | Seed differences | Supported |",
         "|---|---|---|---|---|---|---|---|"]
    names = {"H1": "H1 false completion (FCR, all)", "H2": "H2 escalation (unresolvable)", "H3": "H3 STR (all)",
             "H4": "H4 VTC (resolvable), bound -0.02"}
    for h in ("H1", "H2", "H3", "H4"):
        r = res[h]
        a, b = (r["b"], r["a"]) if h == "H1" else (r["a"], r["b"])
        L.append(f"| {names[h]} | {r['direction']} | RG {_f(a if h != 'H1' else r['b'])} | NG "
                 f"{_f(b if h != 'H1' else r['a'])} | {_c(r)} | {_f(r['p_holm']) if 'p_holm' in r else '—'} | "
                 f"{', '.join(_f(x) for x in r.get('seed_diffs', []))} | {'yes' if r['supported'] else 'no'} |")
    L += ["", f"**RG1 supported overall: {'yes' if res['RG1_supported'] else 'no'}.**", ""]
    if "S1" in res:
        L += ["## S1: RG vs B4 (scripted reference)", "", "| Metric | RG | B4 | RG - B4 [95% CI] |", "|---|---|---|---|"]
        for k in ("vtc_resolvable", "str", "unsafe", "fcr"):
            r = res["S1"][k]
            L.append(f"| {k} | {_f(r['a'])} | {_f(r['b'])} | {_c(r)} |")
        L += ["", "| Group | STR RG | STR B4 | VTC RG | VTC B4 | unsafe RG | unsafe B4 |", "|---|---|---|---|---|---|---|"]
        for g, d in res["S1"]["by_group"].items():
            L.append(f"| {g} | {_f(d['str']['a'])} | {_f(d['str']['b'])} | {_f(d['vtc']['a'])} | {_f(d['vtc']['b'])} | "
                     f"{_f(d['unsafe']['a'])} | {_f(d['unsafe']['b'])} |")
        L.append("")
    if res.get("S2"):
        L += ["## S2: rule ablations (D seed 0; ablation - full gate)", "",
              "| Rule removed | FCR | DMR | unsafe | STR | VTC (resolvable) | escalation (unresolvable) |",
              "|---|---|---|---|---|---|---|"]
        for rule, d in res["S2"].items():
            L.append(f"| {rule} | {_c(d['fcr'])} | {_c(d['dmr'])} | {_c(d['unsafe'])} | {_c(d['str'])} | "
                     f"{_c(d['vtc_resolvable'])} | {_c(d['escalated_unresolvable'])} |")
        L.append("")
    if res.get("S3"):
        L += ["## S3: other agents, gate - no gate", "",
              "| Agent | FCR | unsafe | STR | VTC (resolvable) | escalation (unresolvable) |", "|---|---|---|---|---|---|"]
        for tag, d in res["S3"].items():
            L.append(f"| {tag} | {_c(d['fcr'])} | {_c(d['unsafe'])} | {_c(d['str'])} | {_c(d['vtc_resolvable'])} | "
                     f"{_c(d['escalated_unresolvable'])} |")
        L.append("")
    if "S4" in res:
        L += ["## S4: txn_test replication (descriptive; D's txn_test outcomes were known before RG1)", "",
              "| Metric | Direction | Difference [95% CI] |", "|---|---|---|"]
        for k in ("fcr", "unsafe", "str", "vtc_resolvable", "escalated_unresolvable"):
            r = res["S4"][k]
            L.append(f"| {k} | {'D - RG' if k in ('fcr', 'unsafe') else 'RG - D'} | {_c(r)} |")
        L.append("")
    for sp, tags in res["descriptive"].items():
        if not tags:
            continue
        L += [f"## Descriptive: {sp}", "", "| Tag | episodes | VTC | STR | FCR | DMR | unsafe (unresolvable) | CER | "
              "calls/ep | gate calls/ep |", "|---|---|---|---|---|---|---|---|---|---|"]
        for tag, d in tags.items():
            a = d["all"]
            fr = lambda k: f"{a[k]['num']}/{a[k]['den']}"  # noqa: E731
            L.append(f"| {tag} | {d['n_episodes']} | {fr('VTC')} | {fr('STR')} | {fr('FCR')} | {fr('DMR_all')} | "
                     f"{fr('unsafe_unresolvable')} | {fr('CER')} | {a['cost']['calls']:.2f} | "
                     f"{d['gate_calls_per_episode']:.2f} |")
        L.append("")
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    main()
