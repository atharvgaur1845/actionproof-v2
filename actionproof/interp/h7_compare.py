"""H7 evidence: does the residual stream decode "this task failed" better than the text does?

5-fold cross-validation over task-hash folds, so every episode gets an out-of-fold score from
  - the activation probe: logistic regression at the layer FIXED by the init probe (no layer search on
    held-out folds), and
  - the text baseline: TF-IDF + logistic regression, its config chosen by nested selection inside the
    training folds (text_baseline grid).
Reports both AUROCs and a paired bootstrap CI (resampling episodes) of probe - text.

    python -m actionproof.interp.h7_compare --episodes results/interp/kl/tool_trust/step0.jsonl \
        --npz results/interp/kl/tool_trust/step0.npz --probe results/interp/probe_cues_B.npz
"""
from __future__ import annotations

import argparse
import hashlib
import json

from actionproof.interp.extract import build_examples
from actionproof.interp.text_baseline import prefix_text

GRID = [(an, ng, tail, C) for an, ng in (("word", (1, 2)), ("char_wb", (3, 5)))
        for tail in (1000, 4000) for C in (0.1, 1.0, 10.0)]


def fold_of(task_id: str, k: int) -> int:
    return int(hashlib.sha1(f"h7/{task_id}".encode()).hexdigest(), 16) % k


def text_fit_score(Xtr, ytr, Xte, cfg):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    an, ng, tail, C = cfg
    vec = TfidfVectorizer(analyzer=an, ngram_range=ng, min_df=2, sublinear_tf=True)
    clf = LogisticRegression(C=C, max_iter=3000).fit(vec.fit_transform([x[-tail:] for x in Xtr]), ytr)
    return clf.decision_function(vec.transform([x[-tail:] for x in Xte]))


def compare(episodes, Z, meta, layer: int, pos: str = "B", k: int = 5, boot: int = 2000, seed: int = 0) -> dict:
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.preprocessing import StandardScaler

    index = {(m["task_id"], m["prefix"]): i for i, m in enumerate(meta) if m["pos"] == pos}
    rows = []  # (activation row, text, label, task)
    for r in episodes:
        for ex in build_examples(r):
            if ex["pos"] == pos and (r["task_id"], ex["prefix"]) in index:
                rows.append((index[(r["task_id"], ex["prefix"])],
                             prefix_text(r["conversation"], ex["prefix"], 4000), ex["label"], r["task_id"]))
    H = Z[[r[0] for r in rows], layer].astype(np.float32)
    T = [r[1] for r in rows]
    y = np.array([r[2] for r in rows])
    f = np.array([fold_of(r[3], k) for r in rows])
    s_probe, s_text = np.zeros(len(y)), np.zeros(len(y))
    for j in range(k):
        tr, te = f != j, f == j
        sc = StandardScaler().fit(H[tr])
        s_probe[te] = LogisticRegression(C=0.1, max_iter=3000).fit(sc.transform(H[tr]), y[tr]) \
            .decision_function(sc.transform(H[te]))
        inner = (j + 1) % k  # nested: pick the text config on one training fold
        itr, iva = tr & (f != inner), f == inner
        Ttr = [t for t, m in zip(T, itr) if m]
        Tva = [t for t, m in zip(T, iva) if m]
        best = max(GRID, key=lambda c: roc_auc_score(y[iva], text_fit_score(Ttr, y[itr], Tva, c)))
        s_text[te] = text_fit_score([t for t, m in zip(T, tr) if m], y[tr], [t for t, m in zip(T, te) if m], best)
    a_p, a_t = roc_auc_score(y, s_probe), roc_auc_score(y, s_text)
    rng = np.random.default_rng(seed)
    d = []
    for _ in range(boot):
        b = rng.integers(0, len(y), len(y))
        if len(set(y[b])) == 2:
            d.append(roc_auc_score(y[b], s_probe[b]) - roc_auc_score(y[b], s_text[b]))
    lo, hi = np.percentile(d, [2.5, 97.5])
    return {"pos": pos, "n": int(len(y)), "positives": int(y.sum()), "layer": layer,
            "probe_auroc_cv": float(a_p), "text_auroc_cv": float(a_t),
            "diff": float(a_p - a_t), "diff_ci95": [float(lo), float(hi)]}


def main() -> None:
    import numpy as np
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", required=True)
    ap.add_argument("--npz", required=True)
    ap.add_argument("--probe", default="results/interp/probe_cues_B.npz", help="fixes the layer")
    ap.add_argument("--pos", default="B", choices=["A", "B"])
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    z = np.load(a.npz)
    res = compare([json.loads(l) for l in open(a.episodes)], z["X"], [json.loads(m) for m in z["meta"]],
                  int(np.load(a.probe)["layer"]), a.pos)
    print(json.dumps(res))
    if a.out:
        open(a.out, "w").write(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
