"""Figures for the algorithm-discovery reports, from result files only (run where matplotlib is installed).

  fig_verify_by_site.png   verify-after-write rate by site (order, pay, receive, post) per arm, cw_dev greedy, seed means
  fig_vsc_jc.png           VSC and J_c per arm (seed points + mean) on cw_dev, with the scripted frontier as reference
  fig_training_reads.png   reads per episode over training (T = 1 rollouts), per arm and seed

    python scripts/ad/plot_summary.py --out docs/research/algorithm_discovery/figures
"""
from __future__ import annotations

import argparse
import gzip
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ARMS = ["grpo", "nostd", "awrb", "awrbn", "nostd_ema", "awrbn_ema", "nostd_ema_irc"]
COL = {"grpo": "#4C72B0", "nostd": "#DD8452", "awrb": "#55A868", "awrbn": "#C44E52", "nostd_ema": "#8172B3",
       "awrbn_ema": "#937860", "nostd_ema_irc": "#DA8BC3"}


def summ(tag: str):
    p = ROOT / "results/ad/dev" / f"{tag}__cw_dev.summary.json"
    return json.loads(p.read_text()) if p.exists() else None


def main() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = ROOT / a.out
    out.mkdir(parents=True, exist_ok=True)
    have = {e: [s for s in (summ(f"ad_{e}_s{k}") for k in (0, 1)) if s] for e in ARMS}
    have = {e: v for e, v in have.items() if v}
    sites = ["order", "pay", "receive", "post"]
    fig, ax = plt.subplots(figsize=(9, 3.4))
    w = 0.8 / max(1, len(have))
    for i, (e, ss) in enumerate(have.items()):
        vals = [sum((s["verify_after"].get(x) or [0])[0] or 0 for s in ss) / len(ss) for x in sites]
        ax.bar([j + i * w for j in range(4)], vals, w, label=e, color=COL[e])
    ax.set_xticks([j + 0.4 - w / 2 for j in range(4)])
    ax.set_xticklabels([f"after {x}" + (" (never faulted)" if x == "post" else "") for x in sites])
    ax.set_ylabel("verify rate (cw_dev, greedy)")
    ax.legend(fontsize=7, ncol=4)
    fig.tight_layout()
    fig.savefig(out / "fig_verify_by_site.png", dpi=140)

    fr = json.loads((ROOT / "results/ad/scripted/cw_dev_frontier.json").read_text())["rules"]
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
    for k, (ax, met) in enumerate(zip(axes, ("vsc", "j_c"))):
        for i, (e, ss) in enumerate(have.items()):
            ys = [s[met] for s in ss]
            ax.scatter([i] * len(ys), ys, color=COL[e], s=18)
            ax.hlines(sum(ys) / len(ys), i - 0.3, i + 0.3, color=COL[e])
        for rule, ls in (("never", ":"), ("always", "--"), ("skip_post", "-")):
            ax.axhline(fr[rule]["vsc" if met == "vsc" else "j_c"], color="grey", ls=ls, lw=0.8, label=f"scripted {rule}")
        ax.set_xticks(range(len(have)))
        ax.set_xticklabels(list(have), rotation=30, fontsize=7)
        ax.set_title({"vsc": "VSC (cw_dev)", "j_c": "J_c = R_V - 0.05 reads (cw_dev)"}[met])
    axes[1].legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(out / "fig_vsc_jc.png", dpi=140)

    fig, ax = plt.subplots(figsize=(6.5, 3.4))
    for e in have:
        for k in (0, 1):
            p = ROOT / "results/ad/rollouts" / f"ad_{e}_s{k}.jsonl.gz"
            if not p.exists():
                continue
            by = defaultdict(list)
            with gzip.open(p, "rt") as f:
                for line in f:
                    r = json.loads(line)
                    by[(int(r["step"] or 0)) // 10].append(r["n_verify"])
            xs = sorted(by)
            ax.plot([x * 10 for x in xs], [sum(by[x]) / len(by[x]) for x in xs], "-" if k == 0 else "--", color=COL[e],
                    label=e if k == 0 else None, lw=1)
    ax.set_xlabel("training step")
    ax.set_ylabel("reads per episode (T = 1)")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "fig_training_reads.png", dpi=140)
    print("figures written to", out)


if __name__ == "__main__":
    main()
