"""A12 H3 test exactly as registered (PREREGISTRATION_A12 §7): per fault class, the sign of [read-back at step 200 −
read-back of the warm start] on proc_dev, per arm and seed, against the predicted sign; agreement in >= 80% of
classes with a non-zero prediction.

Read-back per class is the share of episodes that read back a write on the task PO, `a12_gate.reads_back`, the same
definition the predictions were computed with. Signs: a difference within ±1/n of zero is not rounded; exactly 0
is sign "0". Predictions: results/a12b/h3_predictions.json (registered for GRPO under addendum 1); the original
results/a12/h3_predictions.json is reported alongside. Written before the warm-start proc_dev evaluation existed.
    python scripts/h3_test.py --out results/s2/h3_test.json
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ARMS = {"proc_rv": "r_v", "proc_ra": "r_a"}


def _gate():
    spec = importlib.util.spec_from_file_location("a12_gate", ROOT / "scripts/a12_gate.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def per_class(path: Path) -> dict[str, float]:
    reads_back = _gate().reads_back
    by: dict[str, list[bool]] = defaultdict(list)
    for r in map(json.loads, path.open()):
        by[r["fault_class"]].append(reads_back(r))
    return {c: sum(v) / len(v) for c, v in sorted(by.items())}


def sign(x: float) -> str:
    return "+" if x > 0 else "-" if x < 0 else "0"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="0,1,2,3,4")
    ap.add_argument("--out", default="results/s2/h3_test.json")
    a = ap.parse_args()
    warm = per_class(ROOT / "results/eval/a12/dev/warm_start__neutral__proc_dev.jsonl")
    res: dict = {"warm_start_readback": warm, "criterion": ">= 80% sign agreement over classes with a non-zero prediction",
                 "predictions": {}, "runs": {}}
    for label, pf in (("a12b (registered under addendum 1)", "results/a12b/h3_predictions.json"),
                      ("a12 (original gate)", "results/a12/h3_predictions.json")):
        preds = json.loads((ROOT / pf).read_text())["selection"]
        res["predictions"][label] = {}
        for s in a.seeds.split(","):
            for arm, rkey in ARMS.items():
                f = ROOT / f"results/eval/a12/dev/a12_{arm}_s{s}_step200__neutral__proc_dev.jsonl"
                if not f.exists():
                    continue
                rb = per_class(f)
                rows, agree, n = {}, 0, 0
                for c, p in preds[rkey].items():
                    if c not in rb or c not in warm:
                        continue
                    d = rb[c] - warm[c]
                    ok = None if p["predicted_sign"] == "0" else sign(d) == p["predicted_sign"]
                    rows[c] = {"predicted": p["predicted_sign"], "readback_step200": rb[c], "readback_warm": warm[c],
                               "diff": d, "observed": sign(d), "agrees": ok}
                    if ok is not None:
                        n += 1
                        agree += ok
                res["predictions"][label][f"{arm}_s{s}"] = {"classes": rows, "agreement": agree / n if n else None,
                                                            "n_nonzero": n, "passes": n > 0 and agree / n >= 0.8}
    out = ROOT / a.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=2))
    for label, runs in res["predictions"].items():
        for k, v in runs.items():
            print(label[:4], k, "agreement", v["agreement"], "of", v["n_nonzero"], "passes", v["passes"],
                  {c: (r["predicted"], r["observed"]) for c, r in v["classes"].items()})


if __name__ == "__main__":
    main()
