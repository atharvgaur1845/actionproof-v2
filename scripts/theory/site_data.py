"""Chart-ready data for the research site (lead engineer wires it up; see docs/research/SITE_VISUAL_SPEC.md).

    python scripts/theory/site_data.py --out results/theory/site_theory.json      # Slurm (exact enumeration)

Every series is computed exactly from actionproof/theory or copied from a result file; nothing is typed by hand.
Sections:
  phase      learned verify decision over (p, c), canonical problem, G = 8, pi0 = 0.3, for GRPO (shared, iid) and
             Dr. GRPO, plus the Bayes boundary c = p
  update     expected update U(pi = 0.3) vs delta_q for GRPO and Dr. GRPO along c at p = 0.3 (the "stake-free" plot)
  rho        endpoint and regret vs rho (mix, dirmult), p = 0.3, c = 0.05, G = 8 (copied from scan.json)
  cost       learned read probability vs c at p in {0.3, 0.6}, GRPO and Dr. GRPO, shared law (endpoint)
  llm        read share per 10 steps for every logged training run (from results/research/llm_mechanism.json)
  th1        estimator agreement / regret table (from results/theory/th1_heldout_summary.json)
  labels     evidence level of every section
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from actionproof.theory import exact as ex  # noqa: E402
from actionproof.theory.problems import verification  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/theory/site_theory.json")
    a = ap.parse_args()
    ps = [round(x, 3) for x in np.linspace(0.02, 0.98, 25)]
    cs = [round(float(x), 4) for x in np.geomspace(0.005, 0.8, 20)]
    phase = []
    for p in ps:
        for c in cs:
            pb = verification([p], [0.0], [1.0], c=c)
            row = {"p": p, "c": c, "bayes_verify": int(pb.delta_q() > 0)}
            for name, law, est in (("grpo_shared", "shared", "grpo"), ("grpo_iid", "iid", "grpo"),
                                   ("drgrpo", "shared", "nostd")):
                row[name] = ex.endpoint(ex.bernstein(pb, 8, law, est), 0.3)
            phase.append(row)
    update = []
    for c in cs:
        pb = verification([0.3], [0.0], [1.0], c=c)
        update.append({"c": c, "delta_q": pb.delta_q(),
                       "U_grpo": ex.expected_update(pb, 0.3, 8, "shared", "grpo"),
                       "U_drgrpo": ex.expected_update(pb, 0.3, 8, "shared", "nostd")})
    cost = []
    for p in (0.3, 0.6):
        for c in cs:
            pb = verification([p], [0.0], [1.0], c=c)
            cost.append({"p": p, "c": c, "grpo": ex.endpoint(ex.bernstein(pb, 8, "shared", "grpo"), 0.3),
                         "drgrpo": ex.endpoint(ex.bernstein(pb, 8, "shared", "nostd"), 0.3),
                         "bayes": int(pb.delta_q() > 0)})
    scan = json.loads((ROOT / "results/theory/scan.json").read_text())
    rho = [r for r in scan["rho_curves"] if r["p"] == 0.3 and r["c"] == 0.05 and r["G"] == 8 and r["pi0"] == 0.3]
    mech = json.loads((ROOT / "results/research/llm_mechanism.json").read_text())
    llm = {k: {"train_scale": v["info"]["train_scale"], "read_share_by_10_steps": v["read_share_by_10_steps"]}
           for k, v in mech.items() if not k.startswith("_")}
    th1 = json.loads((ROOT / "results/theory/th1_heldout_summary.json").read_text())
    th1t = {e: {"bayes_agreement": v["agrees_bayes_cell"]["rate"], "median_regret": v["median_regret"]}
            for e, v in th1["descriptive"].items()}
    out = {"phase": phase, "update": update, "rho": rho, "cost": cost, "llm": llm, "th1": th1t,
           "labels": {"phase": "analytical (exact)", "update": "analytical (exact)", "rho": "analytical (exact)",
                      "cost": "analytical (exact)", "llm": "language agent, development, descriptive",
                      "th1": "analytical simulation, registered (TH1)"},
           "sources": ["actionproof/theory/exact.py", "results/theory/scan.json",
                       "results/research/llm_mechanism.json", "results/theory/th1_heldout_summary.json"]}
    Path(ROOT / a.out).write_text(json.dumps(out, indent=0))
    print("wrote", a.out, len(phase), "phase cells")


if __name__ == "__main__":
    main()
