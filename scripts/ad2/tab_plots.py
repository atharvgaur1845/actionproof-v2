"""The five required testbed plots, from result files only (run where matplotlib is installed).

    python scripts/ad2/tab_plots.py --dir results/ad2/tabular --out docs/research/algorithm_discovery_v2/figures
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean

SHOW = ["grpo", "grpo_B", "nostd", "rloo", "gdpo", "batchstd", "awrb", "awrbn", "critic_q", "igc", "cvc"]


def load(p: Path) -> list[dict]:
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]


def main() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = load(Path(a.dir) / "TB1.jsonl")
    cell = defaultdict(list)  # (name, p, c, s, L) -> [(vp, vpsafe, voi, J, Jstar)]
    for r in rows:
        _, p, c, s, L, name, seed = r["id"].split("|")
        e = next(iter(r["eval"].values()))
        cell[(name, float(p), float(c), float(s), float(L))].append(
            (e["verify_prob"][0], e["verify_prob"][1], e["voi"][0], e["J"], e["Jstar"], e["regret"]))
    P = sorted({k[1] for k in cell})
    C = sorted({k[2] for k in cell})

    # 1. verify prob vs fault probability (s = 1, L = 0, c = 0.05)
    fig, axs = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for ax, s in zip(axs, (1.0, 0.6)):
        for name in SHOW:
            ys = [mean(x[0] for x in cell[(name, p, 0.05, s, 0.0)]) for p in P if (name, p, 0.05, s, 0.0) in cell]
            ax.plot(P[:len(ys)], ys, marker="o", label=name, lw=2.5 if name in ("grpo", "nostd", "cvc") else 1)
        thr = 0.05 / s
        ax.axvline(thr, color="k", ls="--", lw=1)
        ax.set_title(f"c = 0.05, repair s = {s}; dashed: VOI threshold p = c/s")
        ax.set_xlabel("fault probability p of the risky op")
    axs[0].set_ylabel("learned P(verify | ok ack)")
    axs[1].legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(out / "tb1_verify_vs_p.png", dpi=130)

    # 2. verify prob vs expected VOI (all cells, c >= 0.01)
    fig, axs = plt.subplots(1, 4, figsize=(16, 3.8), sharey=True)
    for ax, name in zip(axs, ("grpo", "nostd", "awrb", "cvc")):
        xs = [(mean(v[2] for v in vals), mean(v[0] for v in vals)) for k, vals in cell.items() if k[0] == name and k[2] >= 0.01]
        ax.scatter([x for x, _ in xs], [y for _, y in xs], s=8, alpha=0.6)
        ax.axvline(0, color="k", lw=1)
        ax.axhline(0.5, color="gray", lw=0.5)
        ax.set_title(name)
        ax.set_xlabel("exact VOI of verifying the risky op")
    axs[0].set_ylabel("learned P(verify)")
    fig.tight_layout()
    fig.savefig(out / "tb1_verify_vs_voi.png", dpi=130)

    # 3. net utility vs read cost (p = 0.2, s = 1, L = 0) and (p = 0.4, s = 0.6)
    fig, axs = plt.subplots(1, 2, figsize=(11, 4))
    for ax, (p, s) in zip(axs, ((0.2, 1.0), (0.4, 0.6))):
        for name in SHOW:
            ys = [mean(x[3] for x in cell[(name, p, c, s, 0.0)]) for c in C]
            ax.plot(C, ys, marker="o", label=name, lw=2.5 if name in ("grpo", "nostd", "cvc") else 1)
        ax.plot(C, [mean(x[4] for x in cell[("nostd", p, c, s, 0.0)]) for c in C], "k--", label="oracle J*")
        ax.set_title(f"p = {p}, s = {s}")
        ax.set_xlabel("read cost c")
        ax.set_ylabel("exact net utility J")
    axs[1].legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(out / "tb1_utility_vs_cost.png", dpi=130)

    # 4. regret vs oracle (TB1 c >= 0.01; TB2 if present)
    names = sorted({k[0] for k in cell})
    reg = {n: mean(x[5] for k, v in cell.items() if k[0] == n and k[2] >= 0.01 for x in v) for n in names}
    fig, ax = plt.subplots(figsize=(10, 4))
    order = sorted(names, key=lambda n: reg[n])
    ax.bar(range(len(order)), [reg[n] for n in order])
    ax.set_xticks(range(len(order)), order, rotation=60, ha="right", fontsize=8)
    ax.set_ylabel("mean regret J* - J (TB1, c >= 0.01)")
    fig.tight_layout()
    fig.savefig(out / "tb1_regret.png", dpi=130)

    # 5. safe vs risky verification rates (TB2 table)
    s2 = Path(a.dir) / "TB2_summary.json"
    if s2.exists():
        tab = json.loads(s2.read_text())["table"]
        keys = [k for k in tab if k.endswith("/x1") or k.endswith("/k1") or k.endswith("/k4")]
        keys = sorted(keys, key=lambda k: tab[k]["regret"])
        fig, ax = plt.subplots(figsize=(12, 4))
        xs = range(len(keys))
        ax.bar([x - 0.2 for x in xs], [tab[k]["risky_verify_rate"] for k in keys], 0.4, label="risky ops")
        ax.bar([x + 0.2 for x in xs], [tab[k]["safe_verify_rate"] for k in keys], 0.4, label="never-faulted op")
        ax.set_xticks(list(xs), keys, rotation=60, ha="right", fontsize=7)
        ax.set_ylabel("P(verify | ok ack)")
        ax.legend()
        ax.set_title("TB2 rich workflows: selective verification (sorted by regret)")
        fig.tight_layout()
        fig.savefig(out / "tb2_safe_vs_risky.png", dpi=130)
    print("figures written to", out)


if __name__ == "__main__":
    main()
