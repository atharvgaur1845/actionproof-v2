"""S3 addendum 4 statistics (PREREGISTRATION_S3_ADDENDUM4): CONTROL (`s3_matched`), ACK (`s3_ackonly`), RSC (`s3_rv_mix`),
seeds 0-4. D1 = RSC - ACK, D2 = ACK - CONTROL on `s3_conf_full` partial-class VSC: one-sided unpaired seed permutation,
Holm over D1 and D2, seed-averaged task bootstrap. Secondary: targeted splits, duplicate class, recovery starts, learners.
    python scripts/s3_stats_a4.py
"""
from __future__ import annotations

import importlib.util
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("s4_stats", ROOT / "scripts/s4_stats.py")
S4 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(S4)
ARMS = {"CONTROL": "s3_matched", "ACK": "s3_ackonly", "RSC": "s3_rv_mix"}
CONTRASTS = {"D1": ("RSC", "ACK"), "D2": ("ACK", "CONTROL")}


def is_partial(v: dict) -> bool:
    return v["cls"] in ("partial", "s3:partial:ack", "s3:partial:read")


def block(split: str, d: str, keep, rng) -> dict:
    A = {n: [S4.load(f"{p}_s{s}", split, d) for s in range(5)] for n, p in ARMS.items()}
    A = {n: [None if m is None else {t: v for t, v in m.items() if keep(v)} for m in ms] for n, ms in A.items()}
    out = {"per_arm_per_seed": {n: [None if m is None else S4.mean(m, "vsc") for m in ms] for n, ms in A.items()},
           "missing": [f"{ARMS[n]}_s{s}__{split}" for n, ms in A.items() for s, m in enumerate(ms) if m is None]}
    for c, (a, b) in CONTRASTS.items():
        if all(A[a]) and all(A[b]):
            out[c] = S4.contrast(A[a], A[b], "vsc", rng, paired=False)
    return out


def main() -> None:
    rng = random.Random(0)
    res = {"primary": block("s3_conf_full", "results/s3/conf", is_partial, rng)}
    ps = {c: res["primary"][c]["unpaired_perm_p_one_sided"] for c in CONTRASTS if c in res["primary"]}
    adj = S4.holm(ps) if ps else {}
    for c in ps:
        r = res["primary"][c]
        r.update(p_holm=adj[c], supported=adj[c] < 0.05 and r["seed_avg"]["ci95"][0] > 0)
    if len(ps) == 2:
        k = (res["primary"]["D1"]["supported"], res["primary"]["D2"]["supported"])
        res["reading"] = {(True, False): "handed evidence", (False, True): "decision-point start",
                          (True, True): "both", (False, False): "inconclusive"}[k]
    res["learners"] = {n: sum(x is not None and x >= 0.40 for x in v)
                       for n, v in res["primary"]["per_arm_per_seed"].items()}
    res["secondary"] = {
        "pp_full": block("s3_pp_full", "results/s3/pp", lambda v: True, rng),
        "pp_rec": block("s3_pp_rec", "results/s3/pp", lambda v: True, rng),
        "dup_full": block("s3_conf_full", "results/s3/conf", lambda v: v["cls"] == "dup_delivery", rng),
        "rec": block("s3_conf_rec", "results/s3/conf", lambda v: True, rng)}
    (ROOT / "results/s3/s3_addendum4.json").write_text(json.dumps(res, indent=2))
    for c in CONTRASTS:
        r = res["primary"].get(c)
        if r:
            print(c, f"{r['seed_avg']['est']:+.3f}", [round(x, 3) for x in r["seed_avg"]["ci95"]],
                  "p", round(r["unpaired_perm_p_one_sided"], 4), "holm", round(r["p_holm"], 4), r["supported"])
    print("reading:", res.get("reading"), "| learners:", res["learners"], "| missing:", res["primary"]["missing"])


if __name__ == "__main__":
    main()
