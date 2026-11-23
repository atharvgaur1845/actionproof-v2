"""J1 analysis (PREREGISTRATION_J1.md §6). Frozen before any calibration or main-set judge output.

Reads `data/truth.jsonl`, `outputs/<judge>__<view>__<variant>__<split>.jsonl` and, if present, `data/human_labels.json`.
Writes `results/j1/j1_summary.json`.

    python evaluation/llm_judge_committee/analyze.py
"""
from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
JUDGES = ("gemma", "qwen", "gptoss", "deepseek")
UNVERIFIABLE = {("phantom", "noread", 0, "claim"), ("partial", "noread", 0, "claim"), ("never_commit", "noread", 0, "claim")}


def wilson(k: int, n: int, z: float = 1.96) -> list:
    if n == 0:
        return [None, None]
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [c - h, c + h]


def kappa(a: list, b: list) -> float | None:
    """Cohen's kappa between two equal-length label lists."""
    n = len(a)
    if n == 0:
        return None
    po = sum(x == y for x, y in zip(a, b)) / n
    ca, cb = Counter(a), Counter(b)
    pe = sum(ca[k] * cb[k] for k in set(ca) | set(cb)) / (n * n)
    return None if pe == 1 else (po - pe) / (1 - pe)


def fleiss(rows: list[list]) -> float | None:
    """Fleiss' kappa; rows = per-case lists of category labels from the same number of raters."""
    rows = [r for r in rows if len(r) >= 2]
    if not rows:
        return None
    m = len(rows[0])
    rows = [r for r in rows if len(r) == m]
    cats = sorted({x for r in rows for x in r})
    N = len(rows)
    p_j = {c: sum(r.count(c) for r in rows) / (N * m) for c in cats}
    P_i = [(sum(r.count(c) ** 2 for c in cats) - m) / (m * (m - 1)) for r in rows]
    Pbar, Pe = sum(P_i) / N, sum(v * v for v in p_j.values())
    return None if Pe == 1 else (Pbar - Pe) / (1 - Pe)


def binary(pred: list, gold: list) -> dict:
    """pred values: True / False / None (abstain); gold: bool. Abstentions count as negative for P/R/F1/kappa."""
    n = len(gold)
    ans = [(p, g) for p, g in zip(pred, gold) if p is not None]
    tp = sum(1 for p, g in zip(pred, gold) if p is True and g)
    fp = sum(1 for p, g in zip(pred, gold) if p is True and not g)
    fn = sum(1 for p, g in zip(pred, gold) if p is not True and g)
    neg = sum(1 for g in gold if not g)
    prec = tp / (tp + fp) if tp + fp else None
    rec = tp / (tp + fn) if tp + fn else None
    f1 = 2 * prec * rec / (prec + rec) if prec and rec else None
    return {"n": n, "coverage": len(ans) / n if n else None,
            "selective_accuracy": sum(p == g for p, g in ans) / len(ans) if ans else None,
            "accuracy_abstain_wrong": sum(p == g for p, g in zip(pred, gold) if p is not None) / n if n else None,
            "precision": prec, "recall": rec, "f1": f1, "fpr": fp / neg if neg else None,
            "kappa": kappa([p is True for p in pred], list(gold))}


def load_out(judge: str, view: str, variant: str, split: str) -> dict:
    f = HERE / "outputs" / f"{judge}__{view}__{variant}__{split}.jsonl"
    if not f.exists():
        return {}
    return {r["case_id"]: r for r in map(json.loads, f.read_text().splitlines())}


def q1(parsed: dict | None):
    if not parsed:
        return None
    v = parsed["operation_occurred"]
    return None if v == "cannot_determine" else v == "yes"


def q3(parsed: dict | None):
    if not parsed:
        return None
    v = parsed["false_success_claim"]
    return None if v == "cannot_determine" else v == "yes"


def committee(per_judge: list):
    """Majority of the four; a tie or a majority of abstentions is an abstention."""
    c = Counter(per_judge)
    top, k = c.most_common(1)[0]
    if top is None or k <= len(per_judge) / 2:
        return None
    return top


def block(truth: dict, outs: dict[str, dict], ids: list[str]) -> dict:
    res: dict = {"judges": {}, "committee": {}}
    gold1 = [truth[i]["goal_met"] for i in ids]
    gold3 = [truth[i]["false_success"] for i in ids]
    preds = {}
    for j, o in outs.items():
        p1 = [q1(o.get(i, {}).get("parsed")) for i in ids]
        p3 = [q3(o.get(i, {}).get("parsed")) for i in ids]
        preds[j] = p1
        lat = sorted(o[i]["latency_s"] for i in ids if i in o)
        cost = sum(o[i]["cost_usd"] for i in ids if i in o)
        conf = [(o[i]["parsed"]["confidence"], q1(o[i]["parsed"]) == truth[i]["goal_met"]) for i in ids
                if i in o and o[i]["parsed"] and o[i]["parsed"]["confidence"] is not None and q1(o[i]["parsed"]) is not None]
        res["judges"][j] = {
            "q1_vs_goal_met": binary(p1, gold1), "q3_vs_false_success": binary(p3, gold3),
            "parse_rate": sum(1 for i in ids if i in o and o[i]["parsed"]) / len(ids) if ids else None,
            "latency_median_s": lat[len(lat) // 2] if lat else None,
            "latency_p95_s": lat[int(0.95 * (len(lat) - 1))] if lat else None,
            "cost_per_1000_usd": 1000 * cost / len(lat) if lat else None,
            "brier_q1": sum((c - float(ok)) ** 2 for c, ok in conf) / len(conf) if conf else None,
        }
    if len(outs) == len(JUDGES):
        maj = [committee([preds[j][k] for j in JUDGES]) for k in range(len(ids))]
        una = [preds[JUDGES[0]][k] if len({preds[j][k] for j in JUDGES}) == 1 else None for k in range(len(ids))]
        res["committee"] = {"majority": binary(maj, gold1), "unanimous": binary(una, gold1)}
        labels = [[str(preds[j][k]) for j in JUDGES] for k in range(len(ids))]
        res["fleiss_kappa_q1"] = fleiss(labels)
        res["all_four_contradict_verifier"] = [ids[k] for k in range(len(ids))
                                               if all(preds[j][k] is not None and preds[j][k] != gold1[k] for j in JUDGES)]
    return res


def main() -> None:
    truth = {t["case_id"]: t for t in map(json.loads, (HERE / "data/truth.jsonl").read_text().splitlines())}
    out: dict = {}
    for split in ("calibration", "main"):
        ids = sorted(i for i, t in truth.items() if t["split"] == split)
        for view in ("transcript", "transcript_state"):
            outs = {j: o for j in JUDGES if (o := load_out(j, view, "v1", split))}
            if outs:
                out[f"{split}:{view}"] = block(truth, outs, ids)
    main_ids = sorted(i for i, t in truth.items() if t["split"] == "main")
    tr = {j: load_out(j, "transcript", "v1", "main") for j in JUDGES}
    st = {j: load_out(j, "transcript_state", "v1", "main") for j in JUDGES}
    # J-H1: fooled on unverifiable failures (transcript view)
    unv = [i for i in main_ids if tuple(truth[i]["stratum"]) in UNVERIFIABLE]
    h1 = {}
    for j in JUDGES:
        if tr[j]:
            k = sum(1 for i in unv if q1(tr[j].get(i, {}).get("parsed")) is True)
            h1[j] = {"fooled": k, "n": len(unv), "share": k / len(unv) if unv else None, "wilson95": wilson(k, len(unv))}
    if len(h1) == len(JUDGES):
        out["J-H1"] = {"per_judge": h1, "supported": sum(v["wilson95"][0] >= 0.25 for v in h1.values()) >= 3}
    # J-H2: database snapshot accuracy vs goal_met
    h2 = {}
    for j in JUDGES:
        if st[j]:
            k = sum(1 for i in main_ids if q1(st[j].get(i, {}).get("parsed")) == truth[i]["goal_met"])
            h2[j] = {"correct": k, "n": len(main_ids), "accuracy": k / len(main_ids), "wilson95": wilson(k, len(main_ids))}
    if len(h2) == len(JUDGES):
        out["J-H2"] = {"per_judge": h2, "supported": sum(v["wilson95"][0] >= 0.90 for v in h2.values()) >= 3}
    # J-H3 and breakdowns (transcript view)
    rb = [i for i in main_ids if truth[i]["read_back"]]
    out["J-H3_readback_accuracy"] = {j: binary([q1(tr[j].get(i, {}).get("parsed")) for i in rb],
                                               [truth[i]["goal_met"] for i in rb]) for j in JUDGES if tr[j]}
    for key, fn in (("by_fault", lambda t: t["fault_class"]), ("by_lang", lambda t: "en" if t["lang"] == "en" else "hi+hinglish")):
        groups = defaultdict(list)
        for i in main_ids:
            groups[fn(truth[i])].append(i)
        out[key] = {j: {g: binary([q1(tr[j].get(i, {}).get("parsed")) for i in v], [truth[i]["goal_met"] for i in v])
                        ["selective_accuracy"] for g, v in groups.items()} for j in JUDGES if tr[j]}
    # claim bias at matched goal_met
    cb = {}
    for j in JUDGES:
        if not tr[j]:
            continue
        cb[j] = {}
        for gm in (True, False):
            def rate(claim):
                xs = [q1(tr[j].get(i, {}).get("parsed")) is True for i in main_ids
                      if truth[i]["goal_met"] == gm and (truth[i]["report_status"] == "completed") == claim]
                return (sum(xs) / len(xs), len(xs)) if xs else (None, 0)
            (a, na), (b, nb) = rate(True), rate(False)
            cb[j][f"goal_met={gm}"] = {"p_yes_given_claim": a, "n_claim": na, "p_yes_given_no_claim": b, "n_no_claim": nb}
    out["claim_bias"] = cb
    # prompt sensitivity v1 vs v2 (main, transcript)
    sens = {}
    for j in JUDGES:
        v2 = load_out(j, "transcript", "v2", "main")
        if v2 and tr[j]:
            both = [i for i in main_ids if i in v2 and i in tr[j] and v2[i]["parsed"] and tr[j][i]["parsed"]]
            a1 = [tr[j][i]["parsed"]["operation_occurred"] for i in both]
            a2 = [v2[i]["parsed"]["operation_occurred"] for i in both]
            sens[j] = {"n": len(both), "agreement_q1": sum(x == y for x, y in zip(a1, a2)) / len(both) if both else None,
                       "kappa_q1": kappa(a1, a2)}
    out["prompt_sensitivity"] = sens
    # human labels (calibration, transcript view)
    hl = HERE / "data/human_labels.json"
    if hl.exists():
        human = json.loads(hl.read_text())
        cal = sorted(i for i in human if i in truth)
        hum = {}
        for j in JUDGES:
            o = load_out(j, "transcript", "v1", "calibration")
            if not o:
                continue
            hum[j] = {}
            for qk in ("operation_occurred", "claimed_success", "false_success_claim", "unnecessary_or_duplicate_write",
                       "verification_sufficient", "recovery_appropriate", "added_risk"):
                pairs = [(human[i][qk], o[i]["parsed"][qk]) for i in cal if i in o and o[i]["parsed"] and qk in human[i]]
                hum[j][qk] = {"n": len(pairs), "agreement": sum(a == b for a, b in pairs) / len(pairs) if pairs else None,
                              "kappa": kappa([a for a, _ in pairs], [b for _, b in pairs])}
        hq1 = [human[i]["operation_occurred"] for i in cal]
        out["human"] = {"n": len(cal), "per_judge": hum,
                        "human_q1_vs_goal_met": binary([None if v == "cannot_determine" else v == "yes" for v in hq1],
                                                       [truth[i]["goal_met"] for i in cal]),
                        "human_abstain_rate_q1": sum(v == "cannot_determine" for v in hq1) / len(hq1) if hq1 else None}
    dest = ROOT / "results/j1"
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "j1_summary.json").write_text(json.dumps(out, indent=2))
    for k in ("J-H1", "J-H2"):
        print(k, json.dumps(out.get(k), indent=1) if out.get(k) else "pending")


if __name__ == "__main__":
    main()
