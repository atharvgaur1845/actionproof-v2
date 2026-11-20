"""A11 H4 analysis (PREREGISTRATION_A11 §9): one annotator, 60 items, between subjects (transcript-only vs
ledger-visible, 10 per stratum x condition). Reports:
  - human false acceptance per condition x stratum (accept = answer "y" on a failed episode, G = 0), true acceptance
    on completions, exact Clopper-Pearson 95% intervals, "u" answers counted separately;
  - the registered prediction: transcript-only false acceptance on invisible failures >= 0.5, ledger-visible <= 0.2;
    falsified if transcript-only false acceptance on invisible failures < 0.3;
  - sensitivity: the same excluding "invisible" items with no faulted write (a benign episode whose failure is a
    visibly skipped action, so the stratum label does not fit);
  - judge vs this annotator per condition (the judge that saw the same view: transcript judges on transcript items,
    grounded judges on ledger items): raw agreement, Cohen's kappa, confusion matrix; judge vs truth.
This is one annotator's reference labels, not human consensus; agreement between annotators is not estimable.
    python scripts/h4_analyze.py --annotator A1 --out results/h4/h4_summary.json
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from scipy.stats import beta

ROOT = Path(__file__).resolve().parents[1]
VIEW_JUDGES = {"transcript": ("aya32b_t__auditkit_tc", "aya32b_t__llm_judge"),
               "ledger": ("aya32b_grounded__auditkit_tc", "aya32b_grounded__llm_judge")}


def cp(k: int, n: int) -> list[float] | None:
    if n == 0:
        return None
    return [0.0 if k == 0 else float(beta.ppf(0.025, k, n - k + 1)), 1.0 if k == n else float(beta.ppf(0.975, k + 1, n - k))]


def rate(rows: list[dict], pred) -> dict:
    n = len(rows)
    k = sum(pred(r) for r in rows)
    return {"k": k, "n": n, "rate": k / n if n else None, "ci95": cp(k, n)}


def kappa(a: list[int], b: list[int]) -> float | None:
    n = len(a)
    if n == 0:
        return None
    po = sum(x == y for x, y in zip(a, b)) / n
    pa, pb = sum(a) / n, sum(b) / n
    pe = pa * pb + (1 - pa) * (1 - pb)
    return None if pe == 1 else (po - pe) / (1 - pe)


def block(rows: list[dict]) -> dict:
    out = {}
    for cond in ("transcript", "ledger"):
        rc = [r for r in rows if r["condition"] == cond]
        out[cond] = {s: {"false_acceptance" if s != "true_completion" else "true_acceptance":
                         rate([r for r in rc if r["stratum"] == s], lambda r: r["answer"] == "y"),
                         "cannot_tell": sum(r["answer"] == "u" for r in rc if r["stratum"] == s)}
                     for s in sorted({r["stratum"] for r in rows})}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--annotator", default="A1")
    ap.add_argument("--out", default="results/h4/h4_summary.json")
    a = ap.parse_args()
    key = {k["item"]: k for k in map(json.loads, (ROOT / "data/h4/key.jsonl").open())}
    labels = [json.loads(line) for line in (ROOT / f"results/h4/labels_{a.annotator}.jsonl").open()]
    if len(labels) != len(key):
        raise SystemExit(f"{len(labels)} of {len(key)} items labelled; finish labelling first")
    rows = [{**key[x["item"]], **x} for x in labels]
    misfit = [r["item"] for r in rows if r["stratum"] == "invisible_failure" and r.get("fault") == "benign"]
    res: dict = {"annotator": a.annotator, "n_items": len(rows), "registered": block(rows),
                 "sensitivity_excluding_benign_invisible": {"excluded_items": misfit,
                                                            **block([r for r in rows if r["item"] not in misfit])}}
    inv_t = res["registered"]["transcript"]["invisible_failure"]["false_acceptance"]
    inv_l = res["registered"]["ledger"]["invisible_failure"]["false_acceptance"]
    res["prediction"] = {"transcript_invisible_fa_ge_0.5": inv_t["rate"] >= 0.5,
                         "ledger_invisible_fa_le_0.2": inv_l["rate"] <= 0.2,
                         "falsified_transcript_invisible_fa_lt_0.3": inv_t["rate"] < 0.3}
    res["judges"] = {}
    for cond, judges in VIEW_JUDGES.items():
        rc = [r for r in rows if r["condition"] == cond and r["answer"] in ("y", "n")]
        for j in judges:
            jr = [r for r in rc if j in r["judges"]]
            human = [int(r["answer"] == "y") for r in jr]
            judge = [int(r["judges"][j] > 0.5) for r in jr]
            truth = [int(r["G"] == 1) for r in jr]
            res["judges"][f"{cond}:{j}"] = {
                "n": len(jr), "raw_agreement_with_annotator": sum(x == y for x, y in zip(human, judge)) / len(jr) if jr else None,
                "kappa_with_annotator": kappa(human, judge),
                "confusion_annotator_x_judge": dict(Counter(f"human={h},judge={g}" for h, g in zip(human, judge))),
                "judge_accuracy_vs_truth": sum(x == y for x, y in zip(judge, truth)) / len(jr) if jr else None,
                "annotator_accuracy_vs_truth": sum(x == y for x, y in zip(human, truth)) / len(jr) if jr else None}
    out = ROOT / a.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=2))
    print(json.dumps({"prediction": res["prediction"], "invisible_transcript": inv_t, "invisible_ledger": inv_l,
                      "misfit_items": misfit}, indent=1))


if __name__ == "__main__":
    main()
