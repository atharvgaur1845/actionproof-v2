"""Figures for paper/main_v4.tex, generated only from committed result files.
  fig_c1_seeds.pdf   : AD-C1 greedy reads/episode vs VSC per seed (results/ad/conf_summaries/*__cw_conf.summary.json)
  fig_th1_laws.pdf   : TH1 Bayes-action agreement by group law and estimator (results/theory/th1_heldout_summary.json)
    python scripts/paper/v4_figures.py
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "paper/figures_v4"
C = ["#257de0", "#f47230", "#00b894", "#f2a600"]  # validated categorical order (dataviz reference palette)
plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": "#888",
                     "axes.linewidth": 0.6, "xtick.color": "#444", "ytick.color": "#444"})


def c1() -> None:
    S = ROOT / "results/ad/conf_summaries"
    arms = [("grpo", "GRPO, 5 seeds = warm start (ring)", C[0], "o"), ("nostd", "std-free", C[1], "s"),
            ("awrb", "AWRB", C[2], "^"), ("awrbn", "AWRB-centred", C[3], "D")]
    fig, ax = plt.subplots(figsize=(3.1, 2.2))
    w = json.loads((S / "ad_base_warm__cw_conf.summary.json").read_text())
    ax.scatter([w["reads"]], [w["vsc"]], s=70, facecolors="none", edgecolors="#555", linewidths=1, zorder=2)
    for arm, lab, col, mk in arms:
        pts = [json.loads((S / f"ad_{arm}_s{k}__cw_conf.summary.json").read_text()) for k in range(5)]
        ax.scatter([p["reads"] for p in pts], [p["vsc"] for p in pts], s=22, color=col, marker=mk, zorder=5 if arm == "grpo" else 3,
                   edgecolors="white", linewidths=0.8)
        x = max(p["reads"] for p in pts); y = sum(p["vsc"] for p in pts) / 5
        ax.annotate(lab, (x, y), xytext=(6, -3) if arm != "grpo" else (8, -11), textcoords="offset points", color="#222", fontsize=7)
    ax.set_xlabel("verification reads per episode (greedy)")
    ax.set_ylabel("verified safe completion")
    ax.set_xlim(-0.15, 3.4); ax.set_ylim(0.655, 0.9)
    ax.grid(axis="y", color="#eee", linewidth=0.6); ax.set_axisbelow(True)
    fig.tight_layout(); fig.savefig(OUT / "fig_c1_seeds.pdf"); plt.close(fig)


def th1() -> None:
    d = json.loads((ROOT / "results/theory/th1_heldout_summary.json").read_text())["descriptive"]
    laws = ["shared", "mix", "dirmult", "iid"]
    ests = [("grpo", "GRPO", C[0]), ("nostd", "Dr. GRPO (centred)", C[1]), ("rloo", "RLOO", C[2])]
    fig, ax = plt.subplots(figsize=(3.1, 2.0))
    bw = 0.26
    for j, (e, lab, col) in enumerate(ests):
        vals = [d[e][f"agrees_bayes|{l}"] for l in laws]
        ax.bar([i + (j - 1) * (bw + 0.02) for i in range(4)], vals, width=bw, color=col, label=lab, zorder=3)
    ax.set_xticks(range(4)); ax.set_xticklabels(["shared", "mix", "Dir.-mult.", "iid"])
    ax.set_ylabel("Bayes-action agreement"); ax.set_ylim(0, 1.05)
    ax.grid(axis="y", color="#eee", linewidth=0.6); ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=6.5, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3, handlelength=1)
    fig.tight_layout(); fig.savefig(OUT / "fig_th1_laws.pdf"); plt.close(fig)


if __name__ == "__main__":
    c1(); th1(); print("ok")
