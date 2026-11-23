"""AD-D2 gates (EXPERIMENT_PROTOCOL_D2.md §4): D1's gates G1-G4 for nostd_ema and awrbn_ema against D1's grpo seeds
0-1, plus the registered descriptive comparisons (AWRB at matched scale; scale effect).

    python scripts/ad/d2_gates.py --out results/ad/d2_summary.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from d1_gates import row  # noqa: E402

CANDS = ("nostd_ema", "awrbn_ema")


def gate(r: list[dict], g: list[dict]) -> dict:
    d = [x["vsc"] - y["vsc"] for x, y in zip(r, g)]
    m = lambda k, xs: sum(x[k] for x in xs) / len(xs)
    kl_ok = all(x["kl100"] is not None and y["kl100"] is not None and x["kl100"] <= 10 * max(y["kl100"], 1e-9)
                for x, y in zip(r, g))
    G = {"G1": sum(d) / 2 >= 0.03 and all(x > 0 for x in d), "G2": m("j_c", r) >= m("j_c", g),
         "G3": m("duplicate", r) <= m("duplicate", g) + 0.02, "G4": kl_ok}
    return {"vsc_diff_per_seed": d, "vsc_diff_mean": sum(d) / 2, "j_c_diff_mean": m("j_c", r) - m("j_c", g),
            "dup_diff_mean": m("duplicate", r) - m("duplicate", g), **G, "pass": all(G.values())}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    runs = {f"ad_{e}_s{s}": row(f"ad_{e}_s{s}") for e in ("grpo", "nostd", "awrbn") + CANDS for s in (0, 1)}
    missing = [k for k, v in runs.items() if v is None]
    gates, desc = {}, {}
    g = [runs[f"ad_grpo_s{s}"] for s in (0, 1)]
    for c in CANDS:
        r = [runs[f"ad_{c}_s{s}"] for s in (0, 1)]
        gates[c] = {"status": "incomplete"} if None in r or None in g else gate(r, g)

    def diff(x, y, k):
        xs, ys = [runs[f"ad_{x}_s{s}"] for s in (0, 1)], [runs[f"ad_{y}_s{s}"] for s in (0, 1)]
        if None in xs or None in ys:
            return None
        return [a_[k] - b_[k] for a_, b_ in zip(xs, ys)] if k != "va_pay" else \
            [(a_["verify_after"]["pay"] or 0) - (b_["verify_after"]["pay"] or 0) for a_, b_ in zip(xs, ys)]
    for name, (x, y) in {"awrb_at_matched_scale": ("awrbn_ema", "nostd_ema"), "scale_effect_raw": ("nostd_ema", "nostd"),
                         "scale_effect_awrb": ("awrbn_ema", "awrbn")}.items():
        desc[name] = {k: diff(x, y, k) for k in ("vsc", "j_c", "va_pay")}
    out = {"runs": runs, "missing": missing, "gates": gates, "descriptive": desc}
    Path(a.out).write_text(json.dumps(out, indent=1))
    print("missing:", missing)
    print(json.dumps({"gates": gates, "descriptive": desc}, indent=1))


if __name__ == "__main__":
    main()
