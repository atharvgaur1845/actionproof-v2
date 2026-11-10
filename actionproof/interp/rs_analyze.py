"""CPU analysis for the reward-source mechanistic study (docs/INTERP_PLAN.md): probes (layer chosen on val), shuffled
and text controls, scenario-held-out probes, M4 knowing-vs-acting, M2 observation-equivalence, C/D on A5, and probe
directions for M3.

    python -m actionproof.interp.rs_analyze --dir results/interp/rs --tags init ack_outcome_s0 ... --out results/interp/rs/analysis.json
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from actionproof.interp.probe import auroc, task_split
from actionproof.interp.rs_extract import GROUPS

RNG = np.random.default_rng(0)


def load(d: Path, tag: str, kinds: str):
    z = np.load(d / f"{tag}__{kinds}.npz")
    return z["X"].astype(np.float32), [int(i) for i in z["ids"]], [json.loads(p) for p in z["probs"]]


def fit(Xtr, ytr):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler().fit(Xtr)
    clf = LogisticRegression(C=1.0, max_iter=2000).fit(sc.transform(Xtr), ytr)
    return sc, clf


def score(m, X):
    sc, clf = m
    return clf.decision_function(sc.transform(X))


def cluster_ci(y, s, groups, n=2000):
    y, s, g = np.asarray(y), np.asarray(s), np.asarray(groups)
    uniq = np.unique(g)
    idx = {u: np.where(g == u)[0] for u in uniq}
    vals = []
    for _ in range(n):
        pick = np.concatenate([idx[u] for u in RNG.choice(uniq, len(uniq))])
        if len(set(y[pick])) == 2:
            vals.append(auroc(y[pick], s[pick]))
    return [round(float(np.percentile(vals, 2.5)), 3), round(float(np.percentile(vals, 97.5)), 3)]


def mean_ci(x, groups, n=2000):
    x, g = np.asarray(x, float), np.asarray(groups)
    uniq = np.unique(g)
    idx = {u: np.where(g == u)[0] for u in uniq}
    vals = [x[np.concatenate([idx[u] for u in RNG.choice(uniq, len(uniq))])].mean() for _ in range(n)]
    return [round(float(np.percentile(vals, 2.5)), 3), round(float(np.percentile(vals, 97.5)), 3)]


def diff_ci(a, ga, b, gb, n=2000):
    a, b, ga, gb = map(np.asarray, (a, b, ga, gb))
    ua, ub = np.unique(ga), np.unique(gb)
    ia, ib = {u: np.where(ga == u)[0] for u in ua}, {u: np.where(gb == u)[0] for u in ub}
    vals = [a[np.concatenate([ia[u] for u in RNG.choice(ua, len(ua))])].mean() -
            b[np.concatenate([ib[u] for u in RNG.choice(ub, len(ub))])].mean() for _ in range(n)]
    return [round(float(np.percentile(vals, 2.5)), 3), round(float(np.percentile(vals, 97.5)), 3)]


def text_auroc(texts, y, tr, te):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    v = TfidfVectorizer(ngram_range=(1, 2)).fit([texts[i] for i in tr])
    clf = LogisticRegression(C=1.0, max_iter=2000).fit(v.transform([texts[i] for i in tr]), [y[i] for i in tr])
    return auroc([y[i] for i in te], clf.decision_function(v.transform([texts[i] for i in te])))


def probe_kind(X, rows, y, held=None):
    """Layer chosen on val; test read once. `held`: (train_mask, test_mask) for scenario-held-out (layer chosen on
    a task-hash val subset of the training portion)."""
    sp = np.array([task_split(r["task_id"]) for r in rows])
    if held is None:
        tr, va, te = sp == "train", sp == "val", sp == "test"
    else:
        trm, tem = held
        tr, va, te = trm & (sp != "val"), trm & (sp == "val"), tem
    y = np.asarray(y)
    val_au = []
    for L in range(X.shape[1]):
        m = fit(X[tr, L], y[tr])
        val_au.append(auroc(y[va], score(m, X[va, L])) if len(set(y[va])) == 2 else float("nan"))
    L = int(np.nanargmax(val_au))
    m = fit(X[tr | va, L] if held is None else X[tr | va, L], y[tr | va])
    s = score(m, X[te, L])
    groups = [rows[i]["task_id"] for i in np.where(te)[0]]
    shuf = y[tr | va].copy(); RNG.shuffle(shuf)
    ms = fit(X[tr | va, L], shuf)
    out = {"layer": L, "n_train": int((tr | va).sum()), "n_test": int(te.sum()),
           "test_auroc": round(auroc(y[te], s), 3), "ci95": cluster_ci(y[te], s, groups),
           "shuffled_label_auroc": round(auroc(y[te], score(ms, X[te, L])), 3),
           "val_auroc_by_layer": [round(v, 3) for v in val_au]}
    texts = [r["conv"][-1]["content"] for r in rows]
    out["text_auroc"] = round(text_auroc(texts, list(y), np.where(tr | va)[0], np.where(te)[0]), 3)
    return out, m, L


def behaviour(rows, probs, y):
    g = {k: np.array([sum(p[t] for t in ts) for p in probs]) for k, ts in GROUPS.items()}
    y = np.asarray(y)
    tasks = [r["task_id"] for r in rows]
    res = {}
    for k, v in g.items():
        pos, neg = v[y == 1], v[y == 0]
        tp, tn = [t for t, yy in zip(tasks, y) if yy == 1], [t for t, yy in zip(tasks, y) if yy == 0]
        res[k] = {"mean_label1": round(float(pos.mean()), 3), "mean_label0": round(float(neg.mean()), 3),
                  "delta": round(float(pos.mean() - neg.mean()), 3), "delta_ci95": diff_ci(pos, tp, neg, tn)}
    return res


def m4(probe, beh):
    dec = probe["ci95"][0] > 0.5
    act = beh["recover"]["delta_ci95"][0] > 0
    return "decodable_and_acted_on" if dec and act else "decodable_but_ignored" if dec else \
        "acted_on_without_linear_decodability" if act else "not_decodable"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="results/interp/rs")
    ap.add_argument("--prefixes", default="results/interp/rs/prefixes.jsonl")
    ap.add_argument("--tags", nargs="+", required=True)
    ap.add_argument("--a5-tag", default="a5_init")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    d = Path(a.dir)
    P = {r["id"]: r for l in open(a.prefixes) if (r := json.loads(l))}
    res: dict = {"models": {}}
    directions = {}
    for tag in a.tags:
        X, ids, probs = load(d, tag, "A1_A2")
        rows = [P[i] for i in ids]
        res["models"][tag] = {}
        for kind in ("A1", "A2"):
            sel = [i for i, r in enumerate(rows) if r["kind"] == kind]
            Xk, rk, pk = X[sel], [rows[i] for i in sel], [probs[i] for i in sel]
            y = [r["label"] for r in rk]
            pr, m, L = probe_kind(Xk, rk, y)
            entry = {"probe": pr, "behaviour": behaviour(rk, pk, y)}
            if kind == "A1":  # observation-equivalence control on v2: F1 ack == landed ack
                f1 = [i for i, r in enumerate(rk) if r["label"] == 0 or r["fault"] == "F1_phantom"]
                entry["probe_F1_only"] = probe_kind(Xk[f1], [rk[i] for i in f1], [y[i] for i in f1])[0]
            if kind == "A2":
                fam = np.array([r["fault"].split("+")[0] if r["label"] == 1 else "neg" for r in rk])
                sp = np.array([task_split(r["task_id"]) for r in rk])
                for held, name in ((("F1_phantom",), "train_F3F2_test_F1"), (("F3_partial", "F2_stale"), "train_F1_test_F3F2")):
                    is_held = np.isin(fam, held)
                    trm = (~is_held & (fam != "neg")) | ((fam == "neg") & (sp != "test"))
                    tem = is_held | ((fam == "neg") & (sp == "test"))
                    try:
                        entry[f"scenario_heldout_{name}"] = probe_kind(Xk, rk, y, held=(trm, tem))[0]
                    except ValueError as e:
                        entry[f"scenario_heldout_{name}"] = {"error": str(e)}
                entry["m4"] = m4(pr, entry["behaviour"])
                sc, clf = m
                w = clf.coef_[0] / sc.scale_
                directions[tag] = {"layer": L, "d": (w / np.linalg.norm(w)).tolist(),
                                   "proj_std": float(np.std(Xk[:, L] @ (w / np.linalg.norm(w))))}
            res["models"][tag][kind] = entry
    # M2: observation-equivalent pairs
    try:
        X, ids, _ = load(d, a.a5_tag, "M2")
        rows = [P[i] for i in ids]
        pairs = defaultdict(dict)
        for i, r in enumerate(rows):
            pairs[(r["pair_id"], r["turn"])][r["label"]] = i
        text_same = act_maxdiff = 0
        full = [(v[0], v[1]) for v in pairs.values() if 0 in v and 1 in v]
        for i, j in full:
            text_same += json.dumps(rows[i]["conv"][1:]) == json.dumps(rows[j]["conv"][1:])
            act_maxdiff = max(act_maxdiff, float(np.abs(X[i] - X[j]).max()))
        y = [r["label"] for r in rows]
        pr, _, _ = probe_kind(X, rows, y)
        res["M2"] = {"pairs_turns": len(full), "identical_text": text_same, "max_abs_activation_diff": act_maxdiff,
                     "hidden_world_probe": {k: pr[k] for k in ("layer", "test_auroc", "ci95", "shuffled_label_auroc",
                                                                "text_auroc")},
                     "leakage_audit_required": bool(text_same < len(full) or act_maxdiff > 0 or pr["ci95"][0] > 0.5)}
    except FileNotFoundError:
        res["M2"] = "not extracted"
    # C/D on A5 (leave-one-family-out, probe vs text)
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        X, ids, _ = load(d, a.a5_tag, "CD")
        rows = [P[i] for i in ids]
        fams = sorted({r["family"] for r in rows})
        texts = [r["conv"][-1]["content"] for r in rows]
        cd = {}
        for lab in ("C", "D"):
            y = np.array([r[lab] for r in rows])
            acc_p, acc_t, n_t = [], [], 0
            L = 20
            for f in fams:
                te = np.array([r["family"] == f for r in rows]); tr = ~te
                if te.sum() == 0 or len(set(y[tr])) < 2:
                    continue
                m = fit(X[tr, L], y[tr]); sc, clf = m
                acc_p.append((clf.predict(sc.transform(X[te, L])) == y[te]).sum())
                v = TfidfVectorizer(ngram_range=(1, 2)).fit([texts[i] for i in np.where(tr)[0]])
                ct = LogisticRegression(C=1.0, max_iter=2000).fit(v.transform([texts[i] for i in np.where(tr)[0]]), y[tr])
                acc_t.append((ct.predict(v.transform([texts[i] for i in np.where(te)[0]])) == y[te]).sum())
                n_t += int(te.sum())
            maj = max((y == c).mean() for c in set(y))
            cd[lab] = {"lofo_probe_acc": round(sum(acc_p) / n_t, 3), "lofo_text_acc": round(sum(acc_t) / n_t, 3),
                       "majority": round(float(maj), 3), "n": int(len(y)), "layer_fixed": L,
                       "classes": {c: int((y == c).sum()) for c in sorted(set(y))}}
        res["CD"] = cd
    except FileNotFoundError:
        res["CD"] = "not extracted"
    Path(a.out).write_text(json.dumps(res, indent=1))
    (Path(a.out).parent / "directions.json").write_text(json.dumps(directions))
    for tag, m in res["models"].items():
        a2, a1 = m["A2"], m["A1"]
        print(f"{tag:16s} A1 auroc {a1['probe']['test_auroc']} (F1-only {a1['probe_F1_only']['test_auroc']}) | "
              f"A2 auroc {a2['probe']['test_auroc']} {a2['probe']['ci95']} text {a2['probe']['text_auroc']} "
              f"shuf {a2['probe']['shuffled_label_auroc']} L{a2['probe']['layer']} | dRecover {a2['behaviour']['recover']['delta']} "
              f"{a2['behaviour']['recover']['delta_ci95']} | P(verify|A1) {a1['behaviour']['verify']['mean_label0']} | {a2['m4']}")
    print("M2", res["M2"]); print("CD", res["CD"])


if __name__ == "__main__":
    main()
