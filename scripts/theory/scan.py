"""Analytical scans for docs/theory (exact, no Monte Carlo). Run on Slurm (slurm/theory_scan.sh).

    python scripts/theory/scan.py --out results/theory/scan.json

Sections (keys of the output):
  threshold_exactness   p* of GRPO when repairability is part of the hidden state (split) vs rollout noise, by c
  cost_blindness        GRPO / Dr. GRPO thresholds over c, shared law, deterministic returns, plus the eps boundary
  iid_interior          endpoint pi* of GRPO under independent states vs G (canonical, s = 1); 1 - pi*, G (1 - pi*)
  critical_rho          rho* above which never-verify is locally stable (mix and dirmult) vs G, with sqrt(G) rho*
  rho_curves            endpoint and regret vs rho (mix, dirmult) for several (p, c), pi0 in {0.3, 0.5}
  severity              K = 3 states: rare catastrophic fault vs common trivial one; GRPO / Dr. GRPO / Bayes
  counterexamples       exact boundary cases (non-binary rewards, state-dependent cost, uninformative reads,
                        false alarms + duplicate penalty, exogenous noise, GDPO, AWRB, multi-component rewards)
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from actionproof.theory import exact as ex  # noqa: E402
from actionproof.theory.problems import rescale_states, verification  # noqa: E402


def thr(make, est, law, G=8, pi=0.3, lo=1e-4, hi=0.9999, tol=1e-6):
    """Smallest fault rate p at which U(pi) > 0 (bisection; None if never; 0 if always)."""
    f = lambda p: ex.expected_update(make(p), pi, G, law, est)
    if f(lo) > 0:
        return 0.0
    if f(hi) <= 0:
        return None
    while hi - lo > tol:
        m = 0.5 * (lo + hi)
        if f(m) > 0:
            hi = m
        else:
            lo = m
    return 0.5 * (lo + hi)


def threshold_exactness():
    out = []
    for s in (1.0, 0.6, 0.3):
        for c in (0.01, 0.05, 0.2):
            row = {"s": s, "c": c, "rule_p_star": 0.5 / s if s > 0 else None, "voi_p_star": c / s}
            for split in (True, False):
                row["grpo_split" if split else "grpo_noise"] = thr(
                    lambda p: verification([p], [0.0], [s], c=c, split_repair=split), "grpo", "shared")
            row["nostd"] = thr(lambda p: verification([p], [0.0], [s], c=c), "nostd", "shared")
            out.append(row)
    return out


def cost_blindness():
    out = []
    for c in (0.0005, 0.001, 0.005, 0.01, 0.05, 0.2, 0.5, 0.9, 0.99):
        for L in (0.0, 1.0):
            out.append({"c": c, "L": L,
                        "grpo": thr(lambda p: verification([p], [L], [1.0], c=c, L0=L), "grpo", "shared"),
                        "nostd": thr(lambda p: verification([p], [L], [1.0], c=c, L0=L), "nostd", "shared"),
                        "voi_p_star": c / (1 + L)})
    return out


def iid_interior():
    out = []
    for p in (0.1, 0.2, 0.3, 0.4, 0.45):
        for c in (0.05, 0.2):
            pb = verification([p], [0.0], [1.0], c=c)
            for G in (2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64):
                b = ex.bernstein(pb, G, "iid", "grpo")
                e = ex.endpoint(b, 0.5)
                out.append({"p": p, "c": c, "G": G, "delta_q": pb.delta_q(), "endpoint_from_0.5": e,
                            "one_minus": 1 - e, "G_one_minus": G * (1 - e), "dU0": G * b[1], "dU1": -G * b[G - 1],
                            "regret": max(pb.J(0), pb.J(1)) - pb.J(e)})
    return out


def critical_rho():
    out = []
    for p in (0.1, 0.2, 0.3, 0.4):
        for c in (0.05, 0.2):
            pb = verification([p], [0.0], [1.0], c=c)
            for G in (2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64):
                bA = ex.bernstein(pb, G, "shared", "grpo")[1]
                bB = ex.bernstein(pb, G, "iid", "grpo")[1]
                rs = None
                if bB > 0 > bA:
                    rs = bB / (bB - bA)
                elif bB <= 0:
                    rs = 0.0
                # dirmult: bisection on rho
                rd = None
                f = lambda r: ex.bernstein(pb, G, f"dirmult:{r}", "grpo")[1]
                if bB <= 0:
                    rd = 0.0
                elif f(0.999) < 0:
                    lo, hi = 1e-4, 0.999
                    for _ in range(40):
                        m = 0.5 * (lo + hi)
                        if f(m) > 0:
                            lo = m
                        else:
                            hi = m
                    rd = 0.5 * (lo + hi)
                out.append({"p": p, "c": c, "G": G, "b1_shared": bA, "b1_iid": bB, "rho_star_mix": rs,
                            "rho_star_dirmult": rd, "sqrtG_rho_mix": None if rs is None else rs * math.sqrt(G)})
    return out


def rho_curves():
    out = []
    rhos = [0.0, 0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    for p, c in ((0.3, 0.05), (0.4, 0.05), (0.2, 0.05), (0.6, 0.05), (0.3, 0.2), (0.15, 0.2)):
        pb = verification([p], [0.0], [1.0], c=c)
        for G in (4, 8, 16):
            for fam in ("mix", "dirmult"):
                for r in rhos:
                    law = "iid" if r == 0 else "shared" if r == 1 else f"{fam}:{r}"
                    b = ex.bernstein(pb, G, law, "grpo")
                    for pi0 in (0.3, 0.5):
                        e = ex.endpoint(b, pi0)
                        out.append({"p": p, "c": c, "G": G, "family": fam, "rho": r, "pi0": pi0, "endpoint": e,
                                    "regret": max(pb.J(0), pb.J(1)) - pb.J(e), "delta_q": pb.delta_q(),
                                    "vote": pb.vote(), "sign_changes": ex.sign_changes(b)})
    return out


def severity():
    """Two fault states: common & mild (p_m, L_m) and rare & catastrophic (p_r, L_r). GRPO's shared-law decision
    depends on the prior masses only; Bayes on the stakes."""
    out = []
    for p_m, p_r in ((0.30, 0.02), (0.40, 0.05), (0.10, 0.01)):
        for L_r in (0.0, 1.0, 5.0, 20.0, 100.0):
            for s_r in (1.0, 0.0):
                pb = verification([p_m, p_r], [0.0, L_r], [0.0, s_r], c=0.05)  # mild fault unrepairable (s=0)
                row = {"p_mild": p_m, "p_rare": p_r, "L_rare": L_r, "s_rare": s_r, "delta_q": pb.delta_q(),
                       "vote": pb.vote()}
                for law in ("shared", "iid", "mix:0.5"):
                    for est in ("grpo", "nostd"):
                        b = ex.bernstein(pb, 8, law, est)
                        e = ex.endpoint(b, 0.3)
                        row[f"{est}|{law}"] = {"endpoint": e, "regret": max(pb.J(0), pb.J(1)) - pb.J(e)}
                out.append(row)
    # invariance witness: rescale the catastrophic state's stakes by 1000
    pb = verification([0.3, 0.02], [0.0, 1.0], [0.0, 1.0], c=0.05)
    pr = rescale_states(pb, [1.0, 1.0, 1000.0])
    wit = {}
    for est in ("grpo", "nostd"):
        wit[est] = [ex.expected_update(pb, 0.3, 8, "shared", est, eps=0.0),
                    ex.expected_update(pr, 0.3, 8, "shared", est, eps=0.0)]
    return {"grid": out, "rescale_witness": wit, "rescaled_delta_q": [pb.delta_q(), pr.delta_q()]}


def counterexamples():
    rows = []

    def rec(name, pb, note, laws=("shared", "iid"), G=8, pi0=0.3, ests=("grpo", "nostd", "gdpo", "awrb", "awrbn")):
        r = {"case": name, "note": note, "delta_q": pb.delta_q(), "vote": pb.vote(),
             "deterministic": pb.deterministic_within_state(), "reads_neutralise": pb.reads_neutralise()}
        for law in laws:
            for est in ests:
                b = ex.bernstein(pb, G, law, est)
                e = ex.endpoint(b, pi0)
                r[f"{est}|{law}"] = {"U_pi0": float(ex.U_from_b(b, pi0)), "endpoint": e,
                                     "regret": max(pb.J(0), pb.J(1)) - pb.J(e)}
        rows.append(r)

    rec("canonical_p0.3", verification([0.3], [0.0], [1.0], c=0.05), "vote -0.4 vs delta_q +0.25")
    rec("canonical_p0.6", verification([0.6], [0.0], [1.0], c=0.05), "vote and delta_q agree (+): GRPO fine")
    rec("canonical_p0.3_c0.4", verification([0.3], [0.0], [1.0], c=0.4), "both negative: GRPO fine")
    rec("cost_flips_outcome_sign", verification([0.6], [0.0], [1.0], c=1.2), "c above the stake: beta_fault < 0")
    rec("state_dependent_cost", verification([0.3], [0.0], [1.0], c=(0.01, 0.4)), "read costs more when faulted")
    rec("state_dependent_cost_b", verification([0.3], [0.0], [1.0], c=(0.4, 0.01)), "read costs more when clean")
    rec("stochastic_repair_noise", verification([0.6], [0.0], [0.6], c=0.05), "s as rollout noise")
    rec("stochastic_repair_state", verification([0.6], [0.0], [0.6], c=0.05, split_repair=True), "s as state")
    rec("imperfect_detection", verification([0.6], [0.0], [1.0], d=[0.7], c=0.05), "misses as rollout noise")
    rec("false_alarm_dup", verification([0.3], [0.0], [1.0], c=0.05, f=0.2, D=1.0),
        "false alarm -> duplicate irreversible action")
    rec("asymmetric_loss", verification([0.3], [5.0], [1.0], c=0.05), "fault loss L = 5 (non-binary stakes)")
    rec("asymmetric_loss_rare", verification([0.1], [20.0], [1.0], c=0.05), "rare, severe")
    rec("exogenous_noise", verification([0.3], [0.0], [1.0], c=0.05, q0=0.3, L0=0.0), "within-state noise")
    rec("uninformative_read", verification([0.3], [0.0], [1.0], c=0.05, uninformative=True), "read reveals nothing")
    rec("multi_fault_K3", verification([0.2, 0.15], [0.0, 3.0], [1.0, 0.5], d=[1.0, 0.8], c=0.05), "K = 3")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/theory/scan.json")
    a = ap.parse_args()
    t0 = time.time()
    res = {}
    for name, fn in (("threshold_exactness", threshold_exactness), ("cost_blindness", cost_blindness),
                     ("iid_interior", iid_interior), ("critical_rho", critical_rho), ("rho_curves", rho_curves),
                     ("severity", severity), ("counterexamples", counterexamples)):
        t = time.time()
        res[name] = fn()
        print(name, round(time.time() - t, 1), "s", flush=True)
    res["seconds"] = round(time.time() - t0, 1)
    out = ROOT / a.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1, default=float))
    print("wrote", out)


if __name__ == "__main__":
    main()
