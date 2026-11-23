"""J2 analysis (PREREGISTRATION_J2.md §5). Frozen before any J2 output.

Reads J1 outputs (`outputs/`), J2 outputs (`outputs_j2/`), `data/truth.jsonl` and, when they exist,
`data/human_labels.json` (author), `data/human_labels_b.json` (second annotator) and
`data/human_labels_adjudicated.json`. Writes `results/j2/j2_summary.json`.

    python evaluation/llm_judge_committee/analyze_j2.py
"""
from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
import sys  # noqa: E402

sys.path.insert(0, str(HERE))
from analyze import binary, fleiss, kappa, wilson  # noqa: E402

J1 = ("gemma", "qwen", "gptoss", "deepseek")
ALL = J1 + ("aya",)
VIEWS = ("transcript", "transcript_state")


def _load(path: Path) -> dict:
    if not path.exists():
        return {}
    return {r["case_id"]: r for r in map(json.loads, path.read_text().splitlines())}


def out(judge: str, view: str, variant: str, tag: str) -> dict:
    if judge == "aya" or tag.startswith("subset"):
        return _load(HERE / "outputs_j2" / f"{judge}__{view}__{variant}__{tag}.jsonl")
    return _load(HERE / "outputs" / f"{judge}__{view}__{variant}__{tag}.jsonl")


def q1(r: dict | None):
    """operation_occurred -> True (yes) / False (partially, no) / None (cannot_determine or unparsed)."""
    if not r or not r.get("parsed"):
        return None
    a = r["parsed"]["operation_occurred"]
    return True if a == "yes" else (None if a == "cannot_determine" else False)


def ans(r: dict | None, key: str):
    if not r or not r.get("parsed"):
        return None
    return r["parsed"][key]


def p_yes(r: dict | None) -> float:
    """Judge's probability that the operation took effect, from its answer and stated confidence."""
    a = q1(r)
    if a is None:
        return 0.5
    c = r["parsed"].get("confidence")
    c = 0.75 if c is None else c
    return c if a else 1 - c


def calib(ps: list[float], gold: list[bool], bins: int = 5) -> dict:
    n = len(ps)
    brier = sum((p - g) ** 2 for p, g in zip(ps, gold)) / n
    ece, rel = 0.0, []
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        idx = [i for i, p in enumerate(ps) if (lo <= p < hi) or (b == bins - 1 and p == 1.0)]
        if idx:
            conf = sum(ps[i] for i in idx) / len(idx)
            acc = sum(gold[i] for i in idx) / len(idx)
            ece += len(idx) / n * abs(conf - acc)
            rel.append({"bin": [lo, hi], "n": len(idx), "mean_p": conf, "frac_true": acc})
    return {"brier": brier, "ece": ece, "reliability": rel}


def agree(a: dict, b: dict, ids: list[str]) -> dict:
    xs = [(q1(a.get(i)), q1(b.get(i))) for i in ids if i in a and i in b]
    if not xs:
        return {"n": 0}
    return {"n": len(xs), "agreement_q1": sum(x == y for x, y in xs) / len(xs),
            "kappa_q1": kappa([str(x) for x, _ in xs], [str(y) for _, y in xs])}


def per_judge(truth: dict, outs: dict, ids: list[str]) -> dict:
    gold = [truth[i]["goal_met"] for i in ids]
    pred = [q1(outs.get(i)) for i in ids]
    m = binary(pred, gold)
    m["false_acceptance"] = m.pop("fpr")  # says the operation took effect when it did not
    m["false_rejection"] = (1 - m["recall"]) if m["recall"] is not None else None  # abstain counts as rejection
    m.update(calib([p_yes(outs.get(i)) for i in ids], gold))
    m.pop("reliability")
    fs = [truth[i]["false_success"] for i in ids]
    a3 = [ans(outs.get(i), "false_success_claim") for i in ids]
    m["q3_false_success"] = binary([None if x in (None, "cannot_determine") else x == "yes" for x in a3], fs)
    dup = [bool(truth[i]["violation"] or truth[i]["duplicate_irreversible"]) for i in ids]
    a4 = [ans(outs.get(i), "unnecessary_or_duplicate_write") for i in ids]
    m["q4_duplicate"] = binary([None if x in (None, "cannot_determine") else x == "yes" for x in a4], dup)
    cl = [truth[i]["report_status"] == "completed" for i in ids]
    m["q2_claimed_success"] = binary([None if ans(outs.get(i), "claimed_success") is None else
                                      ans(outs.get(i), "claimed_success") == "yes" for i in ids], cl)
    rb = [bool(truth[i]["read_back"]) for i in ids]  # proxy for evidence grounding (Q5)
    m["q5_vs_readback_proxy"] = binary([None if ans(outs.get(i), "verification_sufficient") is None else
                                        ans(outs.get(i), "verification_sufficient") == "yes" for i in ids], rb)
    m["parse_rate"] = sum(outs.get(i, {}).get("parsed") is not None for i in ids) / len(ids)
    m["n_missing"] = sum(i not in outs for i in ids)
    m["confident_contradictions"] = [i for i in ids if i in outs and q1(outs[i]) is not None
                                     and q1(outs[i]) != truth[i]["goal_met"]
                                     and (outs[i]["parsed"].get("confidence") or 0) >= 0.9]
    return m


def nb_fit(truth: dict, outs_by_judge: dict, ids: list[str]) -> dict:
    """Error-calibrated weights: per judge, P(answer in {yes, no, abstain} | goal_met) with add-one smoothing,
    fitted on the calibration split against the verifier; class prior from the same split."""
    params = {}
    for j, outs in outs_by_judge.items():
        cnt = {True: Counter(), False: Counter()}
        for i in ids:
            cnt[truth[i]["goal_met"]][str(q1(outs.get(i)))] += 1
        params[j] = {g: {a: (cnt[g][a] + 1) / (sum(cnt[g].values()) + 3) for a in ("True", "False", "None")}
                     for g in (True, False)}
    prior = sum(truth[i]["goal_met"] for i in ids) / len(ids)
    return {"params": params, "prior": prior}


def nb_predict(model: dict, outs_by_judge: dict, i: str) -> float:
    lt = math.log(model["prior"]) - math.log(1 - model["prior"])
    for j, outs in outs_by_judge.items():
        a = str(q1(outs.get(i)))
        pj = model["params"][j]
        lt += math.log(pj[True][a]) - math.log(pj[False][a])
    return 1 / (1 + math.exp(-lt))


def majority(outs_by_judge: dict, i: str):
    v = [q1(o.get(i)) for o in outs_by_judge.values()]
    y, n = sum(x is True for x in v), sum(x is False for x in v)
    return True if y > n else (False if n > y else None)


def aggregate(truth, cal_ids, main_ids, view, judges) -> dict:
    cal = {j: out(j, view, "v1", "calibration") for j in judges}
    mai = {j: out(j, view, "v1", "main") for j in judges}
    if any(not v for v in list(cal.values()) + list(mai.values())):
        return {"status": "missing outputs"}
    model = nb_fit(truth, cal, cal_ids)
    gold = [truth[i]["goal_met"] for i in main_ids]
    pw = [nb_predict(model, mai, i) for i in main_ids]
    maj = [majority(mai, i) for i in main_ids]
    share = [sum(q1(o.get(i)) is True for o in mai.values()) / len(mai) for i in main_ids]
    return {"weighted": {**binary([p > 0.5 for p in pw], gold), **{k: v for k, v in calib(pw, gold).items()
                                                                   if k != "reliability"}},
            "majority": {**binary(maj, gold), **{k: v for k, v in calib(share, gold).items() if k != "reliability"}},
            "weights_log_lr_yes": {j: math.log(model["params"][j][True]["True"] / model["params"][j][False]["True"])
                                   for j in judges},
            "n_cal": len(cal_ids), "n_main": len(main_ids)}


def human(truth, cal_ids, views_out) -> dict:
    res = {}
    files = {"author": "human_labels.json", "second": "human_labels_b.json", "adjudicated": "human_labels_adjudicated.json"}
    labs = {}
    for k, f in files.items():
        p = HERE / "data" / f
        if p.exists():
            labs[k] = json.loads(p.read_text())
    res["available"] = sorted(labs)
    if "author" in labs and "second" in labs:
        ids = [i for i in cal_ids if i in labs["author"] and i in labs["second"]]
        res["inter_annotator_kappa_q1"] = kappa([labs["author"][i].get("operation_occurred") for i in ids],
                                                [labs["second"][i].get("operation_occurred") for i in ids])
        res["inter_annotator_n"] = len(ids)
    ref = labs.get("adjudicated") or labs.get("author")
    if ref:
        res["reference"] = "adjudicated" if "adjudicated" in labs else "author (single annotator, not adjudicated)"
        ids = [i for i in cal_ids if i in ref]
        hum = [ref[i].get("operation_occurred") for i in ids]
        res["human_vs_verifier_q1_agreement"] = sum((h == "yes") == truth[i]["goal_met"] for h, i in zip(hum, ids)) / len(ids)
        for j, o in views_out.items():
            res[f"{j}_kappa_q1_vs_human"] = kappa([ans(o.get(i), "operation_occurred") for i in ids], hum)
            gold = [h == "yes" for h in hum]
            res[f"{j}_q1_vs_human"] = binary([q1(o.get(i)) for i in ids], gold)
    return res


def main() -> None:
    truth = {t["case_id"]: t for t in map(json.loads, (HERE / "data/truth.jsonl").read_text().splitlines())}
    cal_ids = sorted(i for i, t in truth.items() if t["split"] == "calibration")
    main_ids = sorted(i for i, t in truth.items() if t["split"] == "main")
    from committee_j2 import subset_ids
    sub = subset_ids()
    res: dict = {"n": {"calibration": len(cal_ids), "main": len(main_ids), "subset": len(sub)}}
    # 1. per judge, per view, main set vs verifier (Aya added; J1 judges recomputed with the extra metrics)
    for view in VIEWS:
        for j in ALL:
            o = out(j, view, "v1", "main")
            if o:
                res.setdefault("main_vs_verifier", {}).setdefault(view, {})[j] = per_judge(truth, o, main_ids)
    # J-H1 analogue for Aya: fooled on unverifiable failures (transcript view)
    unv = {("phantom", "noread", 0, "claim"), ("partial", "noread", 0, "claim"), ("never_commit", "noread", 0, "claim")}
    ids_u = [i for i in main_ids if tuple(truth[i]["stratum"]) in unv]
    for j in ALL:
        o = out(j, "transcript", "v1", "main")
        if o:
            k = sum(q1(o.get(i)) is True for i in ids_u)
            res.setdefault("fooled_unverifiable_transcript", {})[j] = {"fooled": k, "n": len(ids_u),
                                                                      "share": k / len(ids_u),
                                                                      "wilson95": wilson(k, len(ids_u))}
    # 2. robustness on the subset: order reversal, repeat, paraphrase
    for j in ALL:
        base = out(j, "transcript_state", "v1", "main")
        res.setdefault("order_reversal", {})[j] = agree(base, out(j, "transcript_state", "v1r", "subset"), sub)
        res.setdefault("repeat_run", {})[j] = agree(base, out(j, "transcript_state", "v1", "subset_rep"), sub)
    res["paraphrase_aya"] = agree(out("aya", "transcript", "v1", "main"), out("aya", "transcript", "v2", "subset"), sub)
    # 3. aggregation: error-calibrated weighted vs majority (fit on calibration, evaluate on main)
    for view in VIEWS:
        res.setdefault("aggregation", {})[f"{view}|J1 four"] = aggregate(truth, cal_ids, main_ids, view, J1)
        res["aggregation"][f"{view}|with aya"] = aggregate(truth, cal_ids, main_ids, view, ALL)
    # 4. disagreement: cases where at least one judge says yes and at least one says no
    for view in VIEWS:
        outs = {j: out(j, view, "v1", "main") for j in ALL}
        outs = {j: o for j, o in outs.items() if o}
        split = [i for i in main_ids if {q1(o.get(i)) for o in outs.values()} >= {True, False}]
        res.setdefault("disagreement", {})[view] = {"judges": sorted(outs), "n_split_cases": len(split),
                                                    "share": len(split) / len(main_ids),
                                                    "fleiss_q1": fleiss([[str(q1(o.get(i))) for o in outs.values()]
                                                                         for i in main_ids])}
    # 5. human calibration (blocked until labels exist)
    res["human"] = human(truth, cal_ids, {j: out(j, "transcript", "v1", "calibration") for j in ALL})
    # 6. cost
    tot = 0.0
    for f in (HERE / "outputs_j2").glob("*.jsonl"):
        tot += sum(json.loads(l).get("cost_usd", 0.0) for l in f.read_text().splitlines())
    res["j2_cost_usd_estimate"] = tot
    p = ROOT / "results/j2/j2_summary.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(res, indent=1))
    print(json.dumps({k: res[k] for k in ("n", "fooled_unverifiable_transcript", "order_reversal", "repeat_run",
                                          "paraphrase_aya", "disagreement", "j2_cost_usd_estimate") if k in res},
                     indent=1)[:5000])


if __name__ == "__main__":
    main()
