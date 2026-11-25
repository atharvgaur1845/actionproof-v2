"""W / A / Dw / D 2x2 (research_audit/NOTE_DWARM_EVAL.md). Descriptive; written 2026-11-25 before Dw's output existed.

Factors: recovery SFT (no: W, A; yes: Dw, D) x RS-GRPO recipe (no: W, Dw; yes: A, D). Read rule fixed in the note:
  GRPO increment on SFT       = D - Dw
  GRPO increment on warm start = A - W
  interaction                  = (D - Dw) - (A - W)
  SFT effect without RL        = Dw - W
and, if P3 has reported, Dn - Dw and Dn - D. Paired task bootstrap over the 450-task test union,
B = 10,000, seed 20261125. Units and loaders are those of scripts/audit/p1_decomposition.py (= scripts/p1_stats.py).
All contrasts are EXPLORATORY.

    python scripts/audit/dwarm_2x2.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from p1_decomposition import ROOT, SEEDS, TEST, load  # noqa: E402

ARMS = {
    "W": ("results/p2/ref", ["W"]),
    "A": ("results/p1/test", [f"A_s{s}" for s in SEEDS]),
    "Dw": ("results/p1/test", ["Dw"]),
    "D": ("results/p1/test", [f"D_s{s}" for s in SEEDS]),
    "Dn": ("results/p1/test", [f"Dn_s{s}" for s in SEEDS]),
}
B = 10_000


def main():
    rng = np.random.default_rng(20261125)
    models = {}
    for arm, (dn, tags) in ARMS.items():
        ms = [g[0] for g in (load(dn, t) for t in tags) if g is not None]
        if len(ms) == len(tags):
            models[arm] = ms
        else:
            print(f"arm {arm}: {len(ms)}/{len(tags)} models present; skipped")
    for need in ("W", "A", "Dw", "D"):
        if need not in models:
            raise SystemExit(f"missing arm {need}; the 2x2 cannot be computed")
    mode = "sampled"
    keys = sorted(set.intersection(*(set(m[0][mode]) for m in models.values())))

    def vec(arm, ks):
        return np.array([np.mean([m[mode][k] for m in models[arm]]) for k in ks])

    def contrasts(ks):
        v = {a: vec(a, ks) for a in models}
        defs = {"Dw-W (SFT, no RL)": v["Dw"] - v["W"],
                "A-W (GRPO on warm start)": v["A"] - v["W"],
                "D-Dw (GRPO on recovery SFT)": v["D"] - v["Dw"],
                "interaction (D-Dw)-(A-W)": (v["D"] - v["Dw"]) - (v["A"] - v["W"]),
                "D-A (registered P1-H3, for reference)": v["D"] - v["A"]}
        if "Dn" in v:
            defs["Dn-Dw (no-std GRPO on recovery SFT)"] = v["Dn"] - v["Dw"]
            defs["Dn-D (P3-H1, for reference)"] = v["Dn"] - v["D"]
        out = {}
        for name, d in defs.items():
            bs = d[rng.integers(0, len(d), size=(B, len(d)))].mean(1)
            out[name] = {"delta": float(d.mean()), "ci95": [float(np.quantile(bs, .025)), float(np.quantile(bs, .975))],
                         "n_tasks": int(len(d))}
        return out, {a: float(x.mean()) for a, x in v.items()}

    res = {"status": "EXPLORATORY (descriptive note research_audit/NOTE_DWARM_EVAL.md)", "decoding": mode,
           "n_tasks_union": len(keys)}
    res["contrasts"], res["arm_means"] = contrasts(keys)
    res["per_split"] = {}
    for sp in TEST:
        c, m = contrasts([k for k in keys if k[0] == sp])
        res["per_split"][sp] = {"arm_means": m, "contrasts": c}
    dst = ROOT / "results/audit/dwarm_2x2.json"
    dst.write_text(json.dumps(res, indent=1))
    print(json.dumps({"arm_means": res["arm_means"],
                      "contrasts": {k: [round(v["delta"], 3), [round(x, 3) for x in v["ci95"]]]
                                    for k, v in res["contrasts"].items()}}, indent=1))


if __name__ == "__main__":
    main()
