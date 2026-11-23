"""Figure 1 of main_v2: GRPO's learned verify probability against p*s (the vote variable) and Dr. GRPO's against the
exact VOI, from the raw v2 TB1 file (c >= 0.01).

    python scripts/lab/fig_vote.py   # -> paper/figures_v2/vote.pdf
"""
from __future__ import annotations

import gzip
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    pts = {"grpo": [], "nostd": []}
    with gzip.open(ROOT / "results/ad2/tabular/TB1.jsonl.gz", "rt") as f:
        for line in f:
            r = json.loads(line)
            _, p, c, s, L, name, seed = r["id"].split("|")
            if name in pts and float(c) >= 0.01:
                e = next(iter(r["eval"].values()))
                pts[name].append((float(p) * float(s), e["voi"][0], e["verify_prob"][0], float(c)))
    fig, ax = plt.subplots(1, 2, figsize=(6.3, 2.3), sharey=True)
    cmap = plt.get_cmap("viridis")
    for k, (name, xi, xlabel, vline) in enumerate((("grpo", 0, r"$p\cdot s$ (probability that reading changes the outcome)", 0.5),
                                                   ("nostd", 1, r"exact VOI $= p\,s\,(V{+}L_f)-c$", 0.0))):
        xs = [q[xi] for q in pts[name]]
        ys = [q[2] for q in pts[name]]
        cs = [q[3] for q in pts[name]]
        sc = ax[k].scatter(xs, ys, c=cs, s=6, cmap=cmap, norm=matplotlib.colors.LogNorm(0.01, 0.2), alpha=0.7)
        ax[k].axvline(vline, color="k", lw=0.8)
        ax[k].axhline(0.5, color="0.6", lw=0.5)
        ax[k].set_xlabel(xlabel, fontsize=7)
        ax[k].set_title("GRPO (group std)" if name == "grpo" else "Dr. GRPO (no std)", fontsize=8)
        ax[k].tick_params(labelsize=6)
    ax[0].set_ylabel("learned P(verify)", fontsize=7)
    cb = fig.colorbar(sc, ax=ax, fraction=0.03, pad=0.02)
    cb.set_label("read cost c", fontsize=6)
    cb.ax.tick_params(labelsize=5)
    out = ROOT / "paper/figures_v2"
    out.mkdir(exist_ok=True)
    fig.savefig(out / "vote.pdf", bbox_inches="tight")
    print("wrote", out / "vote.pdf", {k: len(v) for k, v in pts.items()})


if __name__ == "__main__":
    main()
