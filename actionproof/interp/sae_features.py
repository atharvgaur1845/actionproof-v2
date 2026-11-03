"""Tiny Aya SAE features at layer 28 (spec §14, community release Farseen0/tiny-aya-saes).

JumpReLU SAE (SAELens 6.39, apply_b_dec_to_input): f = pre * 1[pre > threshold],
pre = (x - b_dec) @ W_enc + b_enc;  x_hat = f @ W_dec + b_dec.  Hook `model.layers.28`
output = HF hidden_states[29]. Loaded without sae-lens (keeps torch pinned).

Go/no-go: FVU on OUR activations (agent transcripts, thousands of tokens) must be <= 0.3;
the SAE was trained on 256-token CulturaX web text.

    python -m actionproof.interp.sae_features --data data/interp/rft_tiny_train.npz --pos A \
        --out results/interp/sae_A.json
"""
from __future__ import annotations

import argparse
import glob
import json

SAE_GLOB = ("/scratch/pabitra/.cache/huggingface/hub/models--Farseen0--tiny-aya-saes/snapshots/*/"
            "tiny-aya-{variant}/layer_28/sae_weights.safetensors")
HS_INDEX = 29  # hidden_states[0] is the embedding output; layers.28 output is index 29


def load_sae(variant: str = "global"):
    import torch
    from safetensors.torch import load_file
    w = load_file(glob.glob(SAE_GLOB.format(variant=variant))[0])
    return {k: v.float() for k, v in w.items()}


def encode(sae, x):
    pre = (x - sae["b_dec"]) @ sae["W_enc"] + sae["b_enc"]
    return pre * (pre > sae["threshold"])


def decode(sae, f):
    return f @ sae["W_dec"] + sae["b_dec"]


def fvu(x, x_hat) -> float:
    return float(((x - x_hat) ** 2).sum() / ((x - x.mean(0)) ** 2).sum())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--pos", choices=["A", "B"], required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--top", type=int, default=20)
    a = ap.parse_args()

    import numpy as np
    import torch
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score

    from actionproof.interp.probe import task_split

    z = np.load(a.data)
    meta = [json.loads(m) for m in z["meta"]]
    keep = [i for i, m in enumerate(meta) if m["pos"] == a.pos]
    x = torch.tensor(z["X"][keep, HS_INDEX].astype(np.float32))
    meta = [meta[i] for i in keep]
    y = np.array([m["label"] for m in meta])
    sae = load_sae()
    f = encode(sae, x)
    res = {"pos": a.pos, "n": len(y), "fvu": fvu(x, decode(sae, f)),
           "l0": float((f > 0).float().sum(1).mean())}
    F = (f > 0).float().numpy()
    act = f.numpy()
    single = []
    for j in np.where(F.sum(0) >= 5)[0]:
        if len(set(y)) == 2:
            single.append((float(roc_auc_score(y, act[:, j])), int(j), float(F[y == 1, j].mean()), float(F[y == 0, j].mean())))
    single.sort(key=lambda t: -abs(t[0] - 0.5))
    res["top_features"] = [{"feature": j, "auroc": round(s, 4), "fire_rate_pos": round(p, 3), "fire_rate_neg": round(n, 3)}
                           for s, j, p, n in single[: a.top]]
    split = np.array([task_split(m["task_id"]) for m in meta])
    tr, te = split == "train", split == "test"
    if len(set(y[tr])) == 2 and len(set(y[te])) == 2:
        clf = LogisticRegression(penalty="l1", solver="liblinear", C=0.05).fit(act[tr], y[tr])
        res["sparse_probe"] = {"test_auroc": float(roc_auc_score(y[te], clf.decision_function(act[te]))),
                               "nonzero": int((clf.coef_ != 0).sum())}
    res["go"] = res["fvu"] <= 0.3
    with open(a.out, "w") as fh:
        json.dump(res, fh, indent=2)
    print(json.dumps({k: res[k] for k in ("pos", "n", "fvu", "l0", "go")} | {"sparse_probe": res.get("sparse_probe")}))


if __name__ == "__main__":
    main()
