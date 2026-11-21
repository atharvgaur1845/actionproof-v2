"""S3 addendum 2 statistics (PREREGISTRATION_S3_ADDENDUM2).

Part A: RSC (`s3_rv_mix` s0-4) minus the class-matched control (`s3_matched` s0-4) on H3' (`s3_conf_rec`), H4'p and
H4'd (`s3_conf_full`, partial and duplicate-delivery classes): one-sided unpaired seed permutation (252 assignments),
Holm across the three, seed-averaged task bootstrap; descriptive: the registered 3-seed contrast, seeds 3-4 only, a
two-level bootstrap, runs that learn partial repair (partial-class VSC >= 0.40 on `s3_conf_full`).
Part B: receive-then-pay VSC on `s3_pp_full` + `s3_pp_rec` (W), per model and workflow; rebalanced RSC (`s3_rb_mix`
s0-1) and rebalanced control (`s3_rb_matched` s0-1) against RSC and the control seeds 0-1 on the same tasks.
    python scripts/s3_stats_a2.py
"""
from __future__ import annotations

import importlib.util
import json
import random
from pathlib import Path

from actionproof.bench.generator_proc import DATA, load_tasks

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("s4_stats", ROOT / "scripts/s4_stats.py")
S4 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(S4)
RSC, CTL, RV = "s3_rv_mix", "s3_matched", "a12_rv"
RB, RBC = "s3_rb_mix", "s3_rb_matched"
PP = ("s3_pp_full", "s3_pp_rec")
WF = ("order_prepay", "pay_post", "receive_pay")


def subset(d: dict | None, keep) -> dict | None:
    return None if d is None else {t: v for t, v in d.items() if keep(v)}


def models(prefix: str, seeds, split: str, d: str, keep=lambda v: True) -> list[dict | None]:
    return [subset(S4.load(f"{prefix}_s{s}", split, d), keep) for s in seeds]


def part_a(rng: random.Random) -> dict:
    tests = {"H3'": ("s3_conf_rec", lambda v: True),
             "H4'p": ("s3_conf_full", lambda v: v["cls"] == "partial"),
             "H4'd": ("s3_conf_full", lambda v: v["cls"] == "dup_delivery")}
    out: dict = {"tests": {}, "missing": []}
    ps = {}
    for h, (sp, keep) in tests.items():
        rsc = models(RSC, range(5), sp, "results/s3/conf", keep)
        ctl = models(CTL, range(5), sp, "results/s3/conf", keep)
        out["missing"] += [f"{p}_s{s}__{sp}" for p, ms in ((RSC, rsc), (CTL, ctl)) for s, m in enumerate(ms) if m is None]
        r: dict = {"split": sp}
        if all(rsc[:3]) and all(ctl[:3]):
            r["registered_3_seeds"] = S4.contrast(rsc[:3], ctl[:3], "vsc", rng)
        if all(rsc) and all(ctl):
            r["five_seeds"] = S4.contrast(rsc, ctl, "vsc", rng)
            r["seeds_3_4_only"] = S4.contrast(rsc[3:], ctl[3:], "vsc", rng)
            ps[h] = r["five_seeds"]["unpaired_perm_p_one_sided"]
        out["tests"][h] = r
    adj = S4.holm(ps) if ps else {}
    for h in ps:
        f = out["tests"][h]["five_seeds"]
        out["tests"][h].update(p_holm=adj[h], supported=adj[h] < 0.05 and f["seed_avg"]["ci95"][0] > 0)
    learned = {}
    for p, n in ((RSC, 5), (CTL, 5), (RV, 5)):
        ms = models(p, range(n), "s3_conf_full", "results/s3/conf", lambda v: v["cls"] == "partial")
        learned[p] = [None if m is None else round(S4.mean(m, "vsc"), 3) for m in ms]
    out["partial_vsc_full_per_run"] = learned
    out["learns_partial_repair"] = {p: sum(v is not None and v >= 0.40 for v in vs) for p, vs in learned.items()}
    for sp in ("s3_conf_stress", "s3_conf_full"):
        rsc, ctl = models(RSC, range(5), sp, "results/s3/conf"), models(CTL, range(5), sp, "results/s3/conf")
        if all(rsc) and all(ctl):
            out[f"violation_{sp}"] = S4.contrast(rsc, ctl, "violation", rng)
            out[f"vsc_{sp}"] = S4.contrast(rsc, ctl, "vsc", rng) if sp == "s3_conf_stress" else None
    return out


def pp_models(prefix: str, seeds) -> list[dict | None]:
    """Both targeted splits merged, keyed split:task, with each task's workflow."""
    wf = {t.task_id: t.workflow for sp in PP for t in load_tasks(DATA / f"{sp}.jsonl")}
    out = []
    for s in seeds:
        parts = [S4.load(f"{prefix}_s{s}" if s is not None else prefix, sp, "results/s3/pp") for sp in PP]
        if not all(parts):
            out.append(None)
            continue
        out.append({f"{sp}:{t}": {**v, "wf": wf[t]} for sp, d in zip(PP, parts) for t, v in d.items()})
    return out


def part_b(rng: random.Random) -> dict:
    out: dict = {"per_model": {}, "W": {}}
    groups = {"warm_start": [None], RV: range(5), RSC: range(5), CTL: range(5), RB: range(2), RBC: range(2)}
    loaded = {}
    for p, seeds in groups.items():
        ms = pp_models(p, seeds)
        loaded[p] = ms
        for s, m in zip(seeds, ms):
            if m:
                out["per_model"][p if s is None else f"{p}_s{s}"] = {
                    w: S4.mean(subset(m, lambda v, w=w: v["wf"] == w), "vsc") for w in WF}
    for new, old in ((RB, RSC), (RBC, CTL)):
        a = [subset(m, lambda v: v["wf"] == "receive_pay") for m in loaded[new][:2]]
        b = [subset(m, lambda v: v["wf"] == "receive_pay") for m in loaded[old][:2]]
        if all(a) and all(b):
            c = S4.contrast(a, b, "vsc", rng)
            W = sum(c["cand_per_seed"]) / len(c["cand_per_seed"])
            reading = ("frequency" if W >= 0.50 and c["seed_avg"]["ci95"][0] > 0 else
                       "not frequency" if W <= 0.15 else "partly / inconclusive")
            out["W"][new] = {"W": W, "vs": old, "contrast": c, "reading": reading}
        for w in ("order_prepay", "pay_post"):
            a = [subset(m, lambda v, w=w: v["wf"] == w) for m in loaded[new][:2]]
            b = [subset(m, lambda v, w=w: v["wf"] == w) for m in loaded[old][:2]]
            if all(a) and all(b):
                out["W"][f"{new}:{w}"] = S4.contrast(a, b, "vsc", rng)
    return out


def main() -> None:
    rng = random.Random(0)
    res = {"part_a": part_a(rng), "part_b": part_b(rng)}
    (ROOT / "results/s3/s3_addendum2.json").write_text(json.dumps(res, indent=2))
    for h, r in res["part_a"]["tests"].items():
        f = r.get("five_seeds")
        if f:
            print(f"{h:5s} 5 seeds {f['seed_avg']['est']:+.3f} {[round(x, 3) for x in f['seed_avg']['ci95']]} "
                  f"p {f['unpaired_perm_p_one_sided']:.4f} holm {r['p_holm']:.4f} supported {r['supported']}")
    print("learns partial repair:", res["part_a"]["learns_partial_repair"], res["part_a"]["partial_vsc_full_per_run"])
    for k, v in res["part_b"]["per_model"].items():
        print(f"{k:18s}", {w: round(x, 3) for w, x in v.items()})
    for k, v in res["part_b"]["W"].items():
        print(k, v.get("W"), v.get("reading"), v["contrast"]["seed_avg"] if "contrast" in v else v["seed_avg"])
    print("missing:", res["part_a"]["missing"])


if __name__ == "__main__":
    main()
