"""Text-only baseline for the probes (spec §14, PREREGISTRATION H7).

Same examples, same positions, same task-hash split as the activation probe, but the features
are TF-IDF n-grams of the conversation prefix the model saw (system prompt excluded). If text
alone decodes "this task failed" as well as the residual stream does, a high probe AUROC says
nothing about what the model internally registers.

    python -m actionproof.interp.text_baseline --episodes results/interp/kl/tool_trust/step0.jsonl --pos B
"""
from __future__ import annotations

import argparse
import json

from actionproof.interp.extract import build_examples
from actionproof.interp.probe import auroc, task_split


def prefix_text(conv: list[dict], prefix: int, tail_chars: int) -> str:
    text = "\n".join(f"{m['role']}: {m['content']}" for m in conv[1:prefix])
    return text[-tail_chars:]


def run(episodes: list[dict], pos: str, tail_chars: int = 4000) -> dict:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression

    X, y, split = [], [], []
    for r in episodes:
        for ex in build_examples(r):
            if ex["pos"] == pos:
                X.append(prefix_text(r["conversation"], ex["prefix"], tail_chars))
                y.append(ex["label"])
                split.append(task_split(r["task_id"]))
    pick = lambda name, seq: [v for v, s in zip(seq, split) if s == name]
    if any(len(set(pick(n, y))) < 2 for n in ("train", "val", "test")):
        return {"pos": pos, "n": len(y), "text_auroc": None}
    # Give the baseline a fair chance: small grid, chosen on VAL (as the probe's layer is).
    grid = [(an, ng, tail, C) for an, ng in (("word", (1, 2)), ("char_wb", (3, 5)))
            for tail in (1000, tail_chars) for C in (0.1, 1.0, 10.0)]
    best = None
    for an, ng, tail, C in grid:
        Xs = [x[-tail:] for x in X]
        vec = TfidfVectorizer(analyzer=an, ngram_range=ng, min_df=2, sublinear_tf=True)
        clf = LogisticRegression(C=C, max_iter=3000).fit(vec.fit_transform(pick("train", Xs)), pick("train", y))
        va = auroc(pick("val", y), clf.decision_function(vec.transform(pick("val", Xs))))
        if best is None or va > best[0]:
            te = auroc(pick("test", y), clf.decision_function(vec.transform(pick("test", Xs))))
            best = (va, te, {"analyzer": an, "ngram": ng, "tail": tail, "C": C})
    return {"pos": pos, "n": len(y), "n_test": len(pick("test", y)), "positives": sum(y),
            "text_val_auroc": best[0], "text_auroc": best[1], "config": best[2]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", nargs="+", required=True)
    ap.add_argument("--pos", choices=["A", "B"], required=True)
    ap.add_argument("--tail-chars", type=int, default=4000)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    eps = [json.loads(l) for f in a.episodes for l in open(f)]
    res = run(eps, a.pos, a.tail_chars)
    print(json.dumps(res))
    if a.out:
        open(a.out, "w").write(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
