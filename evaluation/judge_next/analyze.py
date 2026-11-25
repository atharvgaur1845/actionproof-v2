"""Metrics for the judge study (results/judge_next/).

    python evaluation/judge_next/analyze.py

**Ground truth.** `data/truth.jsonl`: the deterministic verifier's verdict for natural cases, and the construction
for synthetic cases. "Accept" means completed correctly with no violation.

**Correct decision.** The judge's decision equals the truth. Abstaining is correct only in a view listed in the case's
`abstain_ok_views` (evidence absent from that view). Parse failures count as abstentions.

**Metrics:**
- FAR = P(accept | truth reject); FRR = P(reject | truth accept);
- precision, recall and F1 with accept as the positive class, on covered (non-abstained) cases;
- coverage, abstention rate, and selective risk (error rate on covered cases);
- AUROC and AUPRC from s = P(accept): confidence when accepting, 1 − confidence when rejecting, 0.5 when abstaining
  or when confidence is missing;
- Brier score of s; ECE of the stated confidence against correctness on covered cases (10 bins).

**Gated hybrid (`st_facts_gated`).** The LLM's structured decision with a code gate: any failed verifier check (a step
not verified, an unrefunded duplicate, collateral or unrequested writes) forces reject. Its accuracy is inherited from
the verifier by design. It is reported to show the architecture, not as a judge result.

**Paired comparisons** on identical cases: the difference in correctness with a 10,000-resample case bootstrap
interval, and an exact two-sided McNemar test.
"""
from __future__ import annotations

import json
import math
import random
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = Path(__file__).resolve().parent / "data"
RAW = ROOT / "results/judge_next/raw"
OUT = ROOT / "results/judge_next"


def view_of(cfg: str) -> str:
    return cfg.split("_", 1)[1].replace("facts_gated", "facts")


def auc(pos: list[float], neg: list[float]) -> float | None:
    if not pos or not neg:
        return None
    s = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return s / (len(pos) * len(neg))


def auprc(scores: list[float], labels: list[int]) -> float | None:
    if not any(labels):
        return None
    order = sorted(zip(scores, labels), key=lambda x: -x[0])
    tp = fp = 0
    ap = 0.0
    P = sum(labels)
    for s, y in order:
        if y:
            tp += 1
            ap += tp / (tp + fp)
        else:
            fp += 1
    return ap / P


def metrics(rows: list[dict], truth: dict, cfg: str) -> dict:
    v = view_of(cfg)
    n = len(rows)
    acc_t = [r for r in rows if truth[r["case_id"]]["accept"]]
    rej_t = [r for r in rows if not truth[r["case_id"]]["accept"]]
    dec = lambda r: r["decision"] if r["decision"] in ("accept", "reject") else "abstain"
    correct = lambda r: (dec(r) == ("accept" if truth[r["case_id"]]["accept"] else "reject")) or (
        dec(r) == "abstain" and v in truth[r["case_id"]]["abstain_ok_views"])
    cov = [r for r in rows if dec(r) != "abstain"]
    tp = sum(dec(r) == "accept" and truth[r["case_id"]]["accept"] for r in rows)
    fp = sum(dec(r) == "accept" and not truth[r["case_id"]]["accept"] for r in rows)
    fn = sum(dec(r) == "reject" and truth[r["case_id"]]["accept"] for r in rows)
    prec = tp / (tp + fp) if tp + fp else None
    rec = tp / (tp + fn) if tp + fn else None
    s = []
    for r in rows:
        c = r.get("confidence")
        d = dec(r)
        s.append(0.5 if d == "abstain" or c is None else (c if d == "accept" else 1 - c))
    y = [int(truth[r["case_id"]]["accept"]) for r in rows]
    bins = defaultdict(lambda: [0, 0, 0.0])
    for r in cov:
        c = r.get("confidence")
        if c is None:
            continue
        b = min(9, int(c * 10))
        bins[b][0] += 1
        bins[b][1] += correct(r)
        bins[b][2] += c
    nc = sum(b[0] for b in bins.values())
    ece = sum(abs(b[1] / b[0] - b[2] / b[0]) * b[0] for b in bins.values()) / nc if nc else None
    cm = {"truth_accept": {d: sum(dec(r) == d for r in acc_t) for d in ("accept", "reject", "abstain")},
          "truth_reject": {d: sum(dec(r) == d for r in rej_t) for d in ("accept", "reject", "abstain")}}
    return {"n": n, "accuracy": sum(map(correct, rows)) / n, "parse_rate": sum(r["parsed"] for r in rows) / n,
            "FAR": cm["truth_reject"]["accept"] / len(rej_t) if rej_t else None,
            "FRR": cm["truth_accept"]["reject"] / len(acc_t) if acc_t else None,
            "precision": prec, "recall": rec, "F1": (2 * prec * rec / (prec + rec)) if prec and rec else None,
            "coverage": len(cov) / n, "abstention": 1 - len(cov) / n,
            "selective_risk": (sum(not correct(r) for r in cov) / len(cov)) if cov else None,
            "AUROC": auc([si for si, yi in zip(s, y) if yi], [si for si, yi in zip(s, y) if not yi]),
            "AUPRC": auprc(s, y), "Brier": sum((si - yi) ** 2 for si, yi in zip(s, y)) / n, "ECE": ece,
            "confusion": cm, "_correct": {r["case_id"]: correct(r) for r in rows}}


def gate(rows: list[dict], facts: dict) -> list[dict]:
    out = []
    for r in rows:
        f = facts[r["case_id"]]
        bad = (not f["all_requested_steps_verified"]) or f["unrefunded_duplicate"] or f["collateral_rows"] or f["unrequested_writes"]
        out.append({**r, "decision": "reject" if bad else r["decision"], "parsed": True})
    return out


def mcnemar(b: int, c: int) -> float:
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * p)


def main() -> None:
    truth = {json.loads(l)["case_id"]: json.loads(l) for l in (DATA / "truth.jsonl").read_text().splitlines()}
    cases = {json.loads(l)["case_id"]: json.loads(l) for l in (DATA / "cases.jsonl").read_text().splitlines()}
    facts = {cid: json.loads(c["views"]["facts"].split("## Deterministic verifier facts (mechanically checked; authoritative)\n", 1)[1])
             for cid, c in cases.items()}
    res, per_cat = {}, {}
    for f in sorted(RAW.glob("*.jsonl")):
        tag, cfg = f.stem.split("__")
        rows = [json.loads(l) for l in f.read_text().splitlines()]
        sets = [(cfg, rows)] + ([("st_facts_gated", gate(rows, facts))] if cfg == "st_facts" else [])
        for c, rr in sets:
            m = metrics(rr, truth, c)
            res[f"{tag}|{c}"] = m
            cat = defaultdict(list)
            for r in rr:
                cat[truth[r["case_id"]]["category"]].append(m["_correct"][r["case_id"]])
            per_cat[f"{tag}|{c}"] = {k: sum(v) / len(v) for k, v in sorted(cat.items())}
    ver = [{"case_id": cid, "decision": "accept" if (f["all_requested_steps_verified"] and not (f["unrefunded_duplicate"] or f["collateral_rows"] or f["unrequested_writes"])) else "reject",
            "confidence": 1.0, "parsed": True} for cid, f in facts.items()]
    res["verifier|deterministic_facts"] = metrics(ver, truth, "st_facts")
    rng = random.Random(20261125)
    pairs = []
    keys = sorted(res)
    tags = sorted({k.split("|")[0] for k in keys if not k.startswith("verifier")})
    want = [("v1_state", "v1_transcript"), ("v1_ledger", "v1_transcript"), ("st_state", "v1_state"), ("st_state", "st_transcript"),
            ("st_ledger", "st_transcript"), ("st_facts", "st_state")]
    for t in tags:
        for a, b in want:
            pairs.append((f"{t}|{a}", f"{t}|{b}"))
    for c in ("v1_transcript", "v1_state", "st_state", "st_facts"):
        if len(tags) == 2:
            pairs.append((f"{tags[1]}|{c}", f"{tags[0]}|{c}"))
    comp = {}
    for a, b in pairs:
        if a not in res or b not in res:
            continue
        ca, cb = res[a]["_correct"], res[b]["_correct"]
        ids = sorted(set(ca) & set(cb))
        d = [ca[i] - cb[i] for i in ids]
        boots = sorted(sum(d[rng.randrange(len(d))] for _ in d) / len(d) for _ in range(10000))
        bb = sum(ca[i] and not cb[i] for i in ids)
        cc = sum(cb[i] and not ca[i] for i in ids)
        comp[f"{a} vs {b}"] = {"diff": sum(d) / len(d), "ci95": [boots[249], boots[9749]], "mcnemar_p": mcnemar(bb, cc),
                               "a_only_correct": bb, "b_only_correct": cc, "n": len(ids)}
    for d in ("metrics", "confusion_matrices", "per_fault", "configs"):
        (OUT / d).mkdir(parents=True, exist_ok=True)
    clean = {k: {kk: vv for kk, vv in v.items() if kk not in ("_correct", "confusion")} for k, v in res.items()}
    (OUT / "metrics/metrics.json").write_text(json.dumps(clean, indent=1))
    (OUT / "metrics/paired.json").write_text(json.dumps(comp, indent=1))
    (OUT / "confusion_matrices/confusion.json").write_text(json.dumps({k: v["confusion"] for k, v in res.items()}, indent=1))
    (OUT / "per_fault/per_category.json").write_text(json.dumps(per_cat, indent=1))
    (OUT / "configs/configs.json").write_text(json.dumps({
        "dataset": "evaluation/judge_next/data (build.py, seed 20261125)", "decoding": "greedy, max_tokens 700, vLLM",
        "models": {"aya8b": "CohereLabs/aya-expanse-8b", "aya32b": "CohereLabs/aya-expanse-32b"},
        "configs": {"v1_<view>": "J1 rubric v1 (seven questions); decision rule in run.py", "st_<view>": "structured-evidence prompt (run.py ST_PROMPT)",
                    "st_facts_gated": "st_facts with the verifier gate (analyze.py)", "verifier|deterministic_facts": "decision from verifier facts alone"}}, indent=1))
    f3 = lambda x: "—" if x is None else f"{x:.3f}"
    L = ["# Judge reliability results (generated by evaluation/judge_next/analyze.py; do not edit)", "",
         f"{len(truth)} cases, 17 categories (evaluation/judge_next/build.py). Ground truth: deterministic verifier or construction.", "",
         "| Judge | Config | Accuracy | FAR | FRR | F1 | Coverage | Selective risk | AUROC | AUPRC | Brier | ECE | Parse |", "|" + "---|" * 13]
    for k in keys:
        m = res[k]
        t, c = k.split("|")
        L.append(f"| {t} | {c} | {f3(m['accuracy'])} | {f3(m['FAR'])} | {f3(m['FRR'])} | {f3(m['F1'])} | {f3(m['coverage'])} | {f3(m['selective_risk'])} | "
                 f"{f3(m['AUROC'])} | {f3(m['AUPRC'])} | {f3(m['Brier'])} | {f3(m['ECE'])} | {f3(m['parse_rate'])} |")
    L += ["", "## Paired comparisons (same cases; difference in correctness, case bootstrap, exact McNemar)", "",
          "| Comparison | Diff | 95% CI | McNemar p | a only / b only correct |", "|---|---|---|---|---|"]
    for k, v in comp.items():
        L.append(f"| {k} | {v['diff']:+.3f} | [{v['ci95'][0]:+.3f}, {v['ci95'][1]:+.3f}] | {v['mcnemar_p']:.4f} | {v['a_only_correct']} / {v['b_only_correct']} |")
    cats = sorted({c for v in per_cat.values() for c in v})
    L += ["", "## Accuracy per category", "", "| Judge·config | " + " | ".join(cats) + " |", "|---|" + "---|" * len(cats)]
    for k, v in per_cat.items():
        L.append(f"| {k} | " + " | ".join(f3(v.get(c)) for c in cats) + " |")
    (OUT / "report.md").write_text("\n".join(L) + "\n")
    print("wrote", OUT / "report.md")


if __name__ == "__main__":
    main()
