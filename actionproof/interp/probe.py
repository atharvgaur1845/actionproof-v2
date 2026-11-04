"""Linear probes over layers + text-only baseline (spec §14).

Split BY TASK (no task contributes to both train and test). Per layer: standardised logistic
regression; layer chosen on val AUROC; test AUROC reported overall, per fault and for the F1
negative control. The text baseline (TF-IDF over the observation the model just read) tells us
what the probe adds beyond surface cues.

    python -m actionproof.interp.probe --data data/interp/rft_tiny_train.npz --pos A --out results/interp/probe_A.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict


def task_split(task_id: str) -> str:
    h = int(hashlib.sha1(task_id.encode()).hexdigest(), 16) % 10
    return "train" if h < 6 else ("val" if h < 8 else "test")


def auroc(y, s) -> float:
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(y, s)) if len(set(y)) == 2 else float("nan")


def f1_control(meta: list[dict], scores) -> dict:
    """Negative control (spec §14): a phantom ack is byte-identical to a genuine one, so a probe
    must not separate them. Pairs are compared only WITHIN a stratum (op, k-th call of that op),
    i.e. a Mann-Whitney AUROC over matched pairs: otherwise op identity or "this is a retry"
    (the schedule fires on fixed call indices) can masquerade as signal. The unmatched AUROC is
    kept for comparison."""
    strata = defaultdict(lambda: ([], []))
    flat = ([], [])
    for m, s in zip(meta, scores):
        if m["fault"] not in ("F1_phantom", "none"):
            continue
        g = strata[(m.get("op"), m.get("k"))]
        g[m["fault"] == "F1_phantom"].append(float(s))
        flat[0].append(int(m["fault"] == "F1_phantom"))
        flat[1].append(float(s))
    wins = pairs = 0.0
    for neg, pos in strata.values():
        for p in pos:
            wins += sum((p > n) + 0.5 * (p == n) for n in neg)
            pairs += len(neg)
    return {"f1_control_auroc": wins / pairs if pairs else None, "f1_control_pairs": int(pairs),
            "f1_control_auroc_unmatched": auroc(*flat) if flat[0] else None}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--pos", choices=["A", "B"], required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--C", type=float, default=0.1)
    ap.add_argument("--save-probe", default=None, help="npz with the chosen layer's weights (for guard / #7)")
    a = ap.parse_args()

    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    z = np.load(a.data, allow_pickle=False)
    meta = [json.loads(m) for m in z["meta"]]
    keep = [i for i, m in enumerate(meta) if m["pos"] == a.pos]
    X, meta = z["X"][keep].astype(np.float32), [meta[i] for i in keep]
    y = np.array([m["label"] for m in meta])
    split = np.array([task_split(m["task_id"]) for m in meta])
    tr, va, te = (split == "train"), (split == "val"), (split == "test")

    per_layer = []
    for L in range(X.shape[1]):
        sc = StandardScaler().fit(X[tr, L])
        clf = LogisticRegression(C=a.C, max_iter=2000).fit(sc.transform(X[tr, L]), y[tr])
        per_layer.append({"layer": L, "val_auroc": auroc(y[va], clf.decision_function(sc.transform(X[va, L])))})
    best = max(per_layer, key=lambda r: -1 if np.isnan(r["val_auroc"]) else r["val_auroc"])["layer"]
    sc = StandardScaler().fit(X[tr, best])
    clf = LogisticRegression(C=a.C, max_iter=2000).fit(sc.transform(X[tr, best]), y[tr])
    s_te = clf.decision_function(sc.transform(X[te, best]))

    by_fault = defaultdict(lambda: ([], []))
    for yi, si, m in zip(y[te], s_te, [m for m, t in zip(meta, te) if t]):
        by_fault[m["fault"]][0].append(int(yi))
        by_fault[m["fault"]][1].append(float(si))
    res = {
        "pos": a.pos, "n": {"train": int(tr.sum()), "val": int(va.sum()), "test": int(te.sum())},
        "positives": int(y.sum()), "best_layer": best, "test_auroc": auroc(y[te], s_te),
        "per_layer_val_auroc": [round(r["val_auroc"], 4) for r in per_layer],
        "test_auroc_by_fault": {f: {"n": len(v[0]), "pos": sum(v[0]), "auroc": auroc(*v)} for f, v in by_fault.items()},
    }
    if a.pos == "A":  # F1 control: phantom (label 1) vs genuine acks (label 0) of the same op and call index
        res.update(f1_control([m for m, t in zip(meta, te) if t], s_te))
    # Guard threshold fixed on VAL (never test): the score at which 80% of val positives fire.
    s_va = clf.decision_function(sc.transform(X[va, best]))
    pos_scores = np.sort(s_va[y[va] == 1])
    tau = float(pos_scores[int(0.2 * len(pos_scores))]) if len(pos_scores) else float("inf")
    res["tau80_val"] = tau
    res["val_fire_rate_neg_at_tau"] = float((s_va[y[va] == 0] > tau).mean()) if (y[va] == 0).any() else None
    if a.save_probe:
        np.savez(a.save_probe, layer=best, mean=sc.mean_, scale=sc.scale_, coef=clf.coef_[0],
                 intercept=clf.intercept_, tau=tau)
    with open(a.out, "w") as f:
        json.dump(res, f, indent=2)
    print(json.dumps({k: res[k] for k in ("pos", "n", "best_layer", "test_auroc")} |
                     ({"f1_control_auroc": res.get("f1_control_auroc")} if a.pos == "A" else {})))


if __name__ == "__main__":
    main()
