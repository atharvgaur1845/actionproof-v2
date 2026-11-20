"""Stage 2 paper figures, drawn only from result files (no number typed by hand).
  fig2_a12_seeds       per-seed confirmatory phantom read-back and VSC (with per-seed 95% CI), R_V vs R_A
  fig3_reward_signal   share of informative GRPO groups by fault class, and read rate over training, per arm
  fig4_a14_branches    A14 correct first decisive action per branch, per policy group
    uv run --no-project --with matplotlib python scripts/s2_figures.py --out docs/paper/figs
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RV, RA, WARM, REF = "#0072B2", "#D55E00", "#7F7F7F", "#BBBBBB"
plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6,
                     "xtick.major.width": 0.6, "ytick.major.width": 0.6, "legend.frameon": False})


def fig2(out: Path) -> None:
    d = json.loads((ROOT / "results/s2/a12_summary.json").read_text())
    seeds = d["seeds"]
    fig, ax = plt.subplots(1, 2, figsize=(6.3, 2.2))
    x = range(len(seeds))
    for i, (m, title) in enumerate((("phantom_readback", "Phantom read-back (proc_conf)"), ("vsc", "Verified completion"))):
        ps = d[m]["per_seed"]
        rv = [ps[str(s)]["rv"] for s in seeds]
        ra = [ps[str(s)]["ra"] for s in seeds]
        ax[i].bar([j - 0.18 for j in x], rv, 0.34, color=RV, label="R_V (state)")
        ax[i].bar([j + 0.18 for j in x], ra, 0.34, color=RA, label="R_A (acknowledgement)")
        ax[i].set_xticks(list(x), [f"seed {s}" for s in seeds])
        ax[i].set_ylim(0, 1.05)
        ax[i].set_title(title, fontsize=8)
    fig.legend(*ax[0].get_legend_handles_labels(), loc="upper center", ncol=2, fontsize=7, bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(out / "fig2_a12_seeds.pdf")
    fig.savefig(out / "fig2_a12_seeds.png", dpi=200)


def fig3(out: Path) -> None:
    files = sorted((ROOT / "results/a12").glob("reward_signal_s*.json"))
    classes = ["phantom", "partial", "lost_ack", "never_commit", "clean", "precommit_timeout"]
    colors = ["#0072B2", "#56B4E9", "#D55E00", "#E69F00", "#999999", "#CCCCCC"]
    fig, ax = plt.subplots(1, 3, figsize=(6.3, 2.3), gridspec_kw={"width_ratios": [1, 1, 1.3]})
    for k, (arm, title) in enumerate((("proc_rv", "R_V: informative groups"), ("proc_ra", "R_A: informative groups"))):
        blocks = None
        shares = {c: [] for c in classes}
        for f in files:
            d = json.loads(f.read_text())[arm]
            blocks = sorted(d, key=lambda b: int(b.split("-")[0]))
            for c in classes:
                shares[c].append([d[b].get(c, {}).get("informative_share") or 0 for b in blocks])
        mean = {c: [sum(v[i] for v in shares[c]) / len(shares[c]) for i in range(len(blocks))] for c in classes}
        bottom = [0.0] * len(blocks)
        for c, col in zip(classes, colors):
            ax[k].bar(range(len(blocks)), mean[c], 0.7, bottom=bottom, color=col, label=c.replace("_", " "),
                      edgecolor="white", linewidth=0.5)
            bottom = [b + v for b, v in zip(bottom, mean[c])]
        ax[k].set_xticks(range(len(blocks)), [str(i + 1) for i in range(len(blocks))])
        ax[k].set_xlabel("50-step block")
        ax[k].set_title(title, fontsize=8)
        ax[k].set_ylim(0, 1)
    ax[1].legend(loc="upper left", bbox_to_anchor=(1.0, 1.0), fontsize=6)
    for arm, col, lab in (("proc_rv", RV, "R_V"), ("proc_ra", RA, "R_A")):
        for i, f in enumerate(files):
            d = json.loads(f.read_text())[arm]
            blocks = sorted(d, key=lambda b: int(b.split("-")[0]))
            rate = [sum(r["read_rate"] * r["groups"] for r in d[b].values()) / sum(r["groups"] for r in d[b].values())
                    for b in blocks]
            ax[2].plot([int(b.split("-")[0]) + 25 for b in blocks], rate, color=col, lw=1.2, alpha=0.8,
                       label=lab if i == 0 else None, marker="o", ms=2.5)
    ax[2].set_ylim(0, 1)
    ax[2].set_xlabel("training step")
    ax[2].set_title("Read rate, 5 seeds", fontsize=8)
    ax[2].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "fig3_reward_signal.pdf")
    fig.savefig(out / "fig3_reward_signal.png", dpi=200)


def fig4(out: Path) -> None:
    d = json.loads((ROOT / "results/a14/a14_summary.json").read_text())["policies"]
    branches = [("C", "committed"), ("A", "phantom"), ("L", "lost ack"), ("T", "timeout\nbefore commit"),
                ("P", "partial")]
    groups = [("warm start", ["warm_start"], WARM), ("R_V (3 seeds)", [k for k in d if k.startswith("a12_proc_rv")], RV),
              ("R_A (3 seeds)", [k for k in d if k.startswith("a12_proc_ra")], RA), ("naive (scripted)", ["naive"], REF)]
    fig, ax = plt.subplots(figsize=(6.3, 2.0))
    w = 0.2
    for g, (lab, keys, col) in enumerate(groups):
        for j, (b, _) in enumerate(branches):
            vals = [d[k]["by_branch"][b]["correct"] for k in keys]
            x = j + (g - 1.5) * w
            ax.bar(x, sum(vals) / len(vals), w * 0.9, color=col, label=lab if j == 0 else None)
            if len(vals) > 1:
                ax.plot([x, x], [min(vals), max(vals)], color="black", lw=0.8)
    ax.set_xticks(range(len(branches)), [n for _, n in branches])
    ax.set_ylabel("correct first action")
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=7, ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.22))
    fig.tight_layout()
    fig.savefig(out / "fig4_a14_branches.pdf")
    fig.savefig(out / "fig4_a14_branches.png", dpi=200)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="docs/paper/figs")
    a = ap.parse_args()
    out = ROOT / a.out
    out.mkdir(parents=True, exist_ok=True)
    fig2(out)
    fig3(out)
    fig4(out)
    print("written:", sorted(p.name for p in out.iterdir()))


if __name__ == "__main__":
    main()
