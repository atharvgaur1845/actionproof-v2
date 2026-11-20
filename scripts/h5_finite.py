"""A13 / H5 reported as finite-set results (deviation D-A13-1; `scripts/h5_coverage.py` and its output are frozen
and kept). The registered one-sided Fisher test compared 14/161 G-adv acceptances with 0/13,050 G-rep end states, but
G-rep and G-comb rows are deterministic outcomes repeated over 200 worlds that differ only in ids and amounts, so the
row count is not a sample size. This script reports:
  - G-rep / G-comb as exhaustive enumerations: distinct (workflow, fault plan, defence) configurations, whether any
    configuration's label or acceptance varies across worlds, and how many violating configurations the full contract
    accepts;
  - G-adv as an exact finite test set: counts, every false accept and false reject with its category and ambiguity
    flag, and how the set was produced. No interval or p-value is computed.
    python scripts/h5_finite.py --out results/a13/h5_finite.json
"""
from __future__ import annotations

import argparse
import copy
import importlib.util
import json
from collections import Counter, defaultdict
from pathlib import Path

from actionproof.bench.generator_proc import load_tasks
from actionproof.integrity.clauses import accept
from actionproof.integrity.sim.env import STEP_TOOL
from actionproof.integrity.sim.runtime import episode_record_proc, make_env_proc

ROOT = Path(__file__).resolve().parents[1]


def _mod(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / f"scripts/{name}.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def enumerate_group(two_faults: bool) -> dict:
    h5, h7 = _mod("h5_coverage"), _mod("h7_matrix")
    cells: dict[tuple, set] = defaultdict(set)  # (workflow, plan, defence) -> {(violates, accepted)}
    rows = 0
    for t0 in load_tasks(ROOT / "data/bench/proc/proc_dev.jsonl"):
        req = t0.meta["requested"]
        plans = ([[(s, k)] for s in req for k in h5.KINDS] if not two_faults else
                 [[(a, ka), (b, kb)] for i, a in enumerate(req) for b in req[i + 1:]
                  for ka in h5.KINDS[:4] for kb in h5.KINDS[:4]])
        for plan in plans:
            t = copy.deepcopy(t0)
            never = next((STEP_TOOL[s] for s, k in plan if k == "never_commit"), None)
            t.meta.update(faults=[{"tool": STEP_TOOL[s], "k": 0, "kind": k} for s, k in plan if k != "never_commit"],
                          never_commit=never)
            fam = "+".join(f"{s}:{k}" for s, k in plan)
            for d in h7.DEFENCES:
                env = make_env_proc(t, "h5")
                h7.run_defence(d, env, t)
                rec = episode_record_proc(t, env)
                viol = not rec["score"]["goal_met"]
                ok = accept(rec["task"], rec["task"]["initial_state"], rec["final_state"])
                cells[(t0.workflow, fam, d)].add((viol, ok))
                rows += 1
    varying = sum(len(v) > 1 for v in cells.values())
    viol_cells = [k for k, v in cells.items() if any(x[0] for x in v)]
    accepted = [k for k, v in cells.items() if any(x[0] and x[1] for x in v)]
    return {"rows": rows, "worlds": 200, "configurations": len(cells), "violating_configurations": len(viol_cells),
            "configurations_varying_across_worlds": varying, "violating_configurations_accepted": len(accepted),
            "accepted_examples": [list(k) for k in accepted[:10]]}


def adversarial() -> dict:
    cases = [json.loads(line) for line in (ROOT / "tests/adversarial/proc_adversarial.jsonl").open()]
    fa, fr = [], []
    for c in cases:
        ok = accept(c["task"], c["task"]["initial_state"], c["final_state"])
        row = {"case_id": c["case_id"], "category": c["category"], "ambiguous": bool(c.get("ambiguous")),
               "violated_rule": c.get("violated_rule"), "description": c["description"]}
        if c["violates_semantics"] and ok:
            fa.append(row)
        if not c["violates_semantics"] and not ok:
            fr.append(row)
    viol = [c for c in cases if c["violates_semantics"]]
    unamb = [c for c in viol if not c.get("ambiguous")]
    return {"cases": len(cases), "violations": len(viol), "correct": len(cases) - len(viol),
            "violation_categories": len({c["category"] for c in viol}),
            "false_accepts": len(fa), "false_accepts_unambiguous": sum(not r["ambiguous"] for r in fa),
            "violations_unambiguous": len(unamb), "false_rejects": len(fr),
            "false_accept_categories": dict(Counter(r["category"] for r in fa)),
            "false_accept_cases": fa, "false_reject_cases": fr,
            "provenance": "one author, one deterministic generator (tests/adversarial/gen_proc_adversarial.py, seeds "
                          "1000+i), written from TRANSACTION_SEMANTICS.md and STATE_SCHEMA.md only, without the "
                          "contract, verifier or code; labels (violates_semantics, ambiguous) are the author's; the "
                          "validator was written by the same author. The case mix follows the author's category "
                          "allocation, so the false-accept share describes this set, not a population."}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/a13/h5_finite.json")
    a = ap.parse_args()
    res = {"deviation": "D-A13-1: the registered Fisher test (p = 9.0e-28) treats repeated deterministic rows as "
                        "independent samples and is not used; registered output kept in results/a13/h5_coverage.json",
           "G-rep": enumerate_group(False), "G-comb": enumerate_group(True), "G-adv": adversarial()}
    out = ROOT / a.out
    out.write_text(json.dumps(res, indent=2))
    for g in ("G-rep", "G-comb"):
        print(g, {k: v for k, v in res[g].items() if k != "accepted_examples"})
    print("G-adv", {k: v for k, v in res["G-adv"].items() if not k.endswith("_cases") and k != "provenance"})


if __name__ == "__main__":
    main()
