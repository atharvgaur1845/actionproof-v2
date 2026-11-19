"""A13 / H5: contract coverage generalization (PREREGISTRATION_A13).

Groups of end states, each labelled by an authority that is NOT the clause contract:
  G-rep   single represented faults: proc_dev worlds x requested step x fault kind x scripted defences D0-D5
          (`scripts/h7_matrix.py`); label = the simulator's truth verifier (goal met or not)
  G-comb  two represented faults on two different steps (unseen combinations), same worlds and defences; truth label
  G-adv   independently written adversarial end states (tests/adversarial/proc_adversarial.jsonl); label = the author's
          reading of the plain semantics (violates_semantics), never the contract
Measures: false acceptance (accepts a violation) and false rejection (rejects a correct transaction) of the full
contract; clause necessity (violations a clause alone stops: they pass once it is removed); redundancy (clauses whose
removal changes nothing in any group); pairwise interactions (removing two clauses lets through violations that
neither removal lets through alone); the specification gap (G-adv violations the full contract accepts).
    python scripts/h5_coverage.py --out results/a13/h5_coverage.json
"""
from __future__ import annotations

import argparse
import copy
import itertools
import json
from collections import Counter, defaultdict
from pathlib import Path

from actionproof.bench.generator_proc import load_tasks
from actionproof.integrity.clauses import CLAUSES, accept, failing
from actionproof.integrity.sim.env import STEP_TOOL
from actionproof.integrity.sim.runtime import episode_record_proc, make_env_proc

ROOT = Path(__file__).resolve().parents[1]
KINDS = ("phantom", "partial", "lost_ack", "precommit_timeout", "dup_delivery", "never_commit")


def _h7():
    import importlib.util
    spec = importlib.util.spec_from_file_location("h7", ROOT / "scripts/h7_matrix.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def sim_cases(two_faults: bool) -> list[dict]:
    h7 = _h7()
    out = []
    for t0 in load_tasks(ROOT / "data/bench/proc/proc_dev.jsonl"):
        req = t0.meta["requested"]
        plans = ([[(s, k)] for s in req for k in KINDS] if not two_faults else
                 [[(a, ka), (b, kb)] for a, b in itertools.combinations(req, 2) for ka in KINDS[:4] for kb in KINDS[:4]])
        for plan in plans:
            t = copy.deepcopy(t0)
            never = next((STEP_TOOL[s] for s, k in plan if k == "never_commit"), None)
            t.meta.update(faults=[{"tool": STEP_TOOL[s], "k": 0, "kind": k} for s, k in plan if k != "never_commit"],
                          never_commit=never)
            for d in h7.DEFENCES:
                env = make_env_proc(t, "h5")
                h7.run_defence(d, env, t)
                rec = episode_record_proc(t, env)
                out.append({"family": "+".join(f"{s}:{k}" for s, k in plan), "task": rec["task"],
                            "initial": rec["task"]["initial_state"], "final": rec["final_state"],
                            "violates": not rec["score"]["goal_met"]})
    return out


def adv_cases() -> list[dict]:
    p = ROOT / "tests/adversarial/proc_adversarial.jsonl"
    out = []
    for c in map(json.loads, p.open()):
        out.append({"family": c["category"], "task": c["task"], "initial": c["task"]["initial_state"],
                    "final": c["final_state"], "violates": bool(c["violates_semantics"]), "case_id": c["case_id"],
                    "ambiguous": bool(c.get("ambiguous"))})
    return out


def evaluate(cases: list[dict]) -> dict:
    names = tuple(CLAUSES)
    viol = [c for c in cases if c["violates"]]
    ok = [c for c in cases if not c["violates"]]
    full_fail = [failing(c["task"], c["initial"], c["final"]) for c in viol]
    fa = sum(not f for f in full_fail)
    fr = sum(not accept(c["task"], c["initial"], c["final"]) for c in ok)
    necessity, fam = {}, {}
    for name in names:
        # a violation passes once `name` is removed iff `name` was its only failing clause
        idx = [i for i, f in enumerate(full_fail) if f == [name]]
        necessity[name] = len(idx)
        fam[name] = Counter(viol[i]["family"] for i in idx).most_common(3)
    pair = {}
    for a, b in itertools.combinations(names, 2):
        joint = sum(1 for f in full_fail if f and set(f) <= {a, b} and len(set(f)) == 2)
        if joint:
            pair[f"{a}+{b}"] = joint
    gap = [c.get("case_id", c["family"]) for c, f in zip(viol, full_fail) if not f]
    return {"n": len(cases), "n_violations": len(viol), "n_correct": len(ok),
            "false_acceptance": {"k": fa, "rate": fa / len(viol) if viol else None},
            "false_rejection": {"k": fr, "rate": fr / len(ok) if ok else None},
            "necessity": necessity, "necessity_top_families": fam, "interactions": pair,
            "redundant_here": [n for n in names if necessity[n] == 0], "specification_gap": gap[:200],
            "families": dict(Counter(c["family"] for c in viol))}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/a13/h5_coverage.json")
    a = ap.parse_args()
    groups = {"G-rep": sim_cases(False), "G-comb": sim_cases(True), "G-adv": adv_cases()}
    res = {g: evaluate(c) for g, c in groups.items()}
    adv_unamb = [c for c in groups["G-adv"] if not c.get("ambiguous")]
    res["G-adv (unambiguous only)"] = evaluate(adv_unamb)
    res["redundant_everywhere"] = sorted(set.intersection(*(set(r["redundant_here"]) for r in res.values()
                                                             if isinstance(r, dict) and "redundant_here" in r)))
    # primary test: full-contract false acceptance on G-adv > on G-rep (one-sided Fisher exact)
    from math import comb
    x1, n1 = res["G-adv"]["false_acceptance"]["k"], res["G-adv"]["n_violations"]
    x2, n2 = res["G-rep"]["false_acceptance"]["k"], res["G-rep"]["n_violations"]
    tot, K = n1 + n2, x1 + x2
    p = sum(comb(n1, i) * comb(n2, K - i) for i in range(x1, min(n1, K) + 1)) / comb(tot, K) if K else 1.0
    res["primary_test"] = {"g_adv_fa": [x1, n1], "g_rep_fa": [x2, n2], "fisher_one_sided_p": p}
    out = ROOT / a.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=2))
    print(json.dumps({g: {k: res[g][k] for k in ("n", "n_violations", "false_acceptance", "false_rejection")}
                      for g in groups}, indent=2))
    print(json.dumps(res["primary_test"]), "redundant everywhere:", res["redundant_everywhere"])


if __name__ == "__main__":
    main()
