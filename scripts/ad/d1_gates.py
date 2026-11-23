"""AD-D1 development gates and descriptive mechanism readouts, exactly as registered
(docs/research/algorithm_discovery/EXPERIMENT_PROTOCOL.md §4, §5, Amendment 1). Reads only files written by the
registered tasks; reports missing runs instead of guessing.

    python scripts/ad/d1_gates.py --out results/ad/d1_summary.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEV = ROOT / "results/ad/dev"
TS = ROOT / "results/ad/train_state"
CANDS = ("awrb", "awrbn", "nostd")


def summ(tag: str, split: str = "cw_dev", mode: str = "") -> dict | None:
    p = DEV / f"{tag}__{split}{mode}.summary.json"
    return json.loads(p.read_text()) if p.exists() else None


def kl_at(tag: str, step: int = 100) -> float | None:
    p = TS / f"{tag}.json"
    if not p.exists():
        return None
    rows = [r for r in json.loads(p.read_text()) if "kl" in r and r.get("step") is not None]
    near = [r for r in rows if abs(r["step"] - step) <= 2]
    return sum(r["kl"] for r in near) / len(near) if near else None


def va(s: dict, step: str) -> float | None:
    v = (s.get("verify_after") or {}).get(step)
    return v[0] if v else None


def row(tag: str) -> dict | None:
    s = summ(tag)
    if s is None:
        return None
    samp = [summ(tag, mode=f".t1_s{k}") for k in range(4)]
    samp = [x for x in samp if x]
    pd = summ(tag, "proc_dev")
    return {"vsc": s["vsc"], "r_v": s["r_v"], "j_c": s["j_c"], "reads": s["reads"], "duplicate": s["duplicate"],
            "false_success": s["false_success"], "po_reads": s.get("po_reads"),
            "verify_after": {k: va(s, k) for k in ("order", "pay", "receive", "post")},
            "faulted_vsc": (s.get("faulted") or {}).get("vsc"),
            "sampled_vsc": sum(x["vsc"] for x in samp) / len(samp) if samp else None,
            "sampled_j_c": sum(x["j_c"] for x in samp) / len(samp) if samp else None,
            "proc_dev_vsc": pd["vsc"] if pd else None, "kl100": kl_at(tag)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    runs = {}
    for est in ("grpo",) + CANDS:
        for s in (0, 1):
            runs[f"ad_{est}_s{s}"] = row(f"ad_{est}_s{s}")
    for t in ("ad_awrbc30_s0", "ad_awrbnc30_s0", "ad_base_warm", "ad_base_a12rv_s0", "ad_base_rsc_s0"):
        runs[t] = row(t)
    missing = [k for k, v in runs.items() if v is None]
    gates = {}
    g = [runs[f"ad_grpo_s{s}"] for s in (0, 1)]
    for c in CANDS:
        r = [runs[f"ad_{c}_s{s}"] for s in (0, 1)]
        if None in r or None in g:
            gates[c] = {"status": "incomplete"}
            continue
        d = [x["vsc"] - y["vsc"] for x, y in zip(r, g)]
        mean = lambda k, xs: sum(x[k] for x in xs) / len(xs)
        kl_ok = all(x["kl100"] is not None and y["kl100"] is not None and x["kl100"] <= 10 * max(y["kl100"], 1e-9)
                    for x, y in zip(r, g))
        G = {"G1": sum(d) / 2 >= 0.03 and all(x > 0 for x in d), "G2": mean("j_c", r) >= mean("j_c", g),
             "G3": mean("duplicate", r) <= mean("duplicate", g) + 0.02, "G4": kl_ok}
        gates[c] = {"vsc_diff_per_seed": d, "vsc_diff_mean": sum(d) / 2,
                    "j_c_diff_mean": mean("j_c", r) - mean("j_c", g),
                    "dup_diff_mean": mean("duplicate", r) - mean("duplicate", g), **G, "pass": all(G.values())}
    out = {"runs": runs, "missing": missing, "gates": gates,
           "note": "development data, two seeds; gates are a development screen, not a statistical test"}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=1))
    for k, v in runs.items():
        if v:
            print(k, {x: (round(y, 3) if isinstance(y, float) else y) for x, y in v.items() if x != "verify_after"},
                  {x: (round(y, 2) if y is not None else None) for x, y in v["verify_after"].items()})
    print("missing:", missing)
    print(json.dumps(gates, indent=1))


if __name__ == "__main__":
    main()
