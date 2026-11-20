"""A11 H1 analysis (PREREGISTRATION_A11 §7-8): Delta_policy, Delta_runtime, Delta_information per cell, with paired
task bootstrap CIs (10,000 resamples, seed 0, the same task indices for every quantity in a contrast), and the
registered interaction I = Delta_policy(Enforce) - Delta_policy(Passthrough) for each trained seed separately.

Inputs: results/a11/eval/<tag>__<mode>__<split>.jsonl (system level VSC = record["system_level"]["vsc"]);
taxonomy labels data/bench/c2/fault_labels.jsonl. Missing files are reported as missing, never filled.
    python scripts/a11_estimands.py --out results/s2/a11_summary.json
"""
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPLITS = ("test_iid_c2", "v2_stress_c2")
MODES = ("passthrough", "observe", "enforce", "enforce_inform")
WARM = "warm_start"
TRAINED = ("pcstr_d_s0", "pcstr_d_s1", "ack_outcome_s0", "ack_outcome_s1", "tool_trust_s0")
REFERENCE = ("aya32b_prompt",)
B = 10_000


def load(tag: str, mode: str) -> dict[str, float] | None:
    out = {}
    for sp in SPLITS:
        p = ROOT / f"results/a11/eval/{tag}__{mode}__{sp}.jsonl"
        if not p.exists():
            return None
        for r in map(json.loads, p.open()):
            out[r["task_id"]] = float(r["system_level"]["vsc"])
    return out


def boot(fn, tasks: list[str], rng: random.Random) -> tuple[float, list[float]]:
    est = fn(tasks)
    vals = sorted(fn([tasks[rng.randrange(len(tasks))] for _ in tasks]) for _ in range(B))
    return est, [vals[int(0.025 * B)], vals[int(0.975 * B) - 1]]


def mean(d: dict[str, float], ts: list[str]) -> float:
    return sum(d[t] for t in ts) / len(ts)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/s2/a11_summary.json")
    a = ap.parse_args()
    labels = {r["task_id"]: r for r in map(json.loads, (ROOT / "data/bench/c2/fault_labels.jsonl").open())}
    data = {(t, m): load(t, m) for t in (WARM,) + TRAINED + REFERENCE for m in MODES}
    missing = sorted(f"{t}__{m}" for (t, m), v in data.items() if v is None)
    cells = defaultdict(list)
    for tid, lab in labels.items():
        cells[lab["cell"]].append(tid)
    cells["ALL"] = sorted(labels)
    cells["ALL_excl_F5"] = sorted(t for t, lb in labels.items() if "F5" not in lb.get("faults", []))
    res: dict = {"missing": missing, "cells": {c: len(v) for c, v in cells.items()}, "estimands": {}, "interaction": {}}
    for c, ts in cells.items():
        rng = random.Random(0)
        e = res["estimands"].setdefault(c, {})
        for pol in TRAINED:
            for m in MODES:
                if data[(pol, m)] and data[(WARM, m)]:
                    d1, d0 = data[(pol, m)], data[(WARM, m)]
                    v, ci = boot(lambda s: mean(d1, s) - mean(d0, s), ts, rng)
                    e[f"delta_policy[{pol},{m}]"] = {"est": v, "ci95": ci}
        for pol in (WARM,) + TRAINED + REFERENCE:
            ob, en, inf = data[(pol, "observe")], data[(pol, "enforce")], data[(pol, "enforce_inform")]
            if ob and en:
                v, ci = boot(lambda s: mean(en, s) - mean(ob, s), ts, rng)
                e[f"delta_runtime[{pol}]"] = {"est": v, "ci95": ci}
            if en and inf:
                v, ci = boot(lambda s: mean(inf, s) - mean(en, s), ts, rng)
                e[f"delta_information[{pol}]"] = {"est": v, "ci95": ci}
            for m in MODES:
                if data[(pol, m)]:
                    e[f"vsc[{pol},{m}]"] = mean(data[(pol, m)], ts)
    full = [t for t, lb in labels.items() if lb["state_observability"] == "Full"]
    rng = random.Random(0)
    for pol in TRAINED:
        p_e, w_e, p_p, w_p = (data[(pol, "enforce")], data[(WARM, "enforce")],
                              data[(pol, "passthrough")], data[(WARM, "passthrough")])
        if all((p_e, w_e, p_p, w_p)):
            v, ci = boot(lambda s: (mean(p_e, s) - mean(w_e, s)) - (mean(p_p, s) - mean(w_p, s)), full, rng)
            res["interaction"][pol] = {"I": v, "ci95": ci, "rejects_no_interaction": not (ci[0] <= 0 <= ci[1]),
                                       "role": "primary" if pol.startswith("pcstr_d") else "secondary"}
    pr = [res["interaction"].get(p) for p in ("pcstr_d_s0", "pcstr_d_s1")]
    res["primary_claim"] = (None if None in pr else bool(all(x["rejects_no_interaction"] for x in pr) and
                            (pr[0]["I"] > 0) == (pr[1]["I"] > 0)))
    out = ROOT / a.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=2))
    print("missing:", missing)
    print(json.dumps(res["interaction"], indent=1), "primary claim:", res["primary_claim"])


if __name__ == "__main__":
    main()
