"""Article figures from result files (run where the raw sampled episodes are: the cluster copy).

  F1 docs/figures/f1_channel_verification.png  post-write refund verification by reward channel, three warm starts
  F3 docs/figures/f3_before_after.png          VSC on the untouched confirmatory splits, warm start vs trained arms
  docs/figures/figures_data.json               every plotted value with its source file (the table view)

F1 rates: v2 warm start = share of episodes with a success-acked refund that read it back (`a4_metrics.pwv`), from
the sampled T=1 x 8 val episodes in results/eval/a4/h11/; A6 and A7 = greedy test_iid `pwv_issue_refund.rate` from
the evaluation summaries. Sampled and greedy rates are not comparable across panels, only within one.

    .venv/bin/python scripts/make_figures.py
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from actionproof.eval.a4_metrics import pwv  # noqa: E402

OUT = Path("docs/figures")
STATE, ACK, INK, MUTED, SURF = "#2a78d6", "#eb6834", "#0b0b0b", "#898781", "#fcfcfb"
plt.rcParams.update({"font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": "#52514e", "xtick.color": "#52514e",
                     "ytick.color": "#52514e", "axes.spines.top": False, "axes.spines.right": False,
                     "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF})


def wilson(p: float, n: int) -> tuple[float, float]:
    z = 1.96
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def sampled_pwv(path: str) -> tuple[float, int]:
    v = [pwv(json.loads(l)["trace"], "issue_refund") for l in open(path)]
    v = [x for x in v if x is not None]
    return sum(v) / len(v), len(v)


def greedy_pwv(path: str) -> tuple[float, int]:
    a = json.load(open(path))["a4"]["pwv_issue_refund"]
    return a["rate"], a.get("n", 0)


def f1(data: list) -> None:
    H, E = "results/eval/a4/h11", "results/eval/a6"
    panels = [
        ("v2 warm start\nsampled, val", [
            ("warm start", None, f"{H}/init__val_T1x8.jsonl", sampled_pwv),
            ("outcome s0", "state", f"{H}/outcome_s0__val_T1x8.jsonl", sampled_pwv),
            ("PCSTR-d s0", "state", f"{H}/pcstr_d_s0__val_T1x8.jsonl", sampled_pwv),
            ("ack_outcome s0", "ack", f"{H}/ack_outcome_s0__val_T1x8.jsonl", sampled_pwv),
            ("ack_outcome s1", "ack", f"{H}/ack_outcome_s1__val_T1x8.jsonl", sampled_pwv),
            ("tool_trust s0", "ack", f"{H}/tool_trust_s0__val_T1x8.jsonl", sampled_pwv)]),
        ("A6 warm start\ngreedy, test-IID", [
            ("warm start", None, f"{E}/init_a6__neutral__test_iid.summary.json", greedy_pwv),
            *[(a, "state", f"{E}/a6_{a}__neutral__test_iid.summary.json", greedy_pwv)
              for a in ("outcome_s0", "pcstr_d_s0", "pcstr_d_noreport_s0", "pcstr_d_noreport_s1", "pcstr_d_noND_s0",
                        "pcstr_d_noFunder_s0")],
            ("ack_outcome s0", "ack", f"{E}/a6_ack_outcome_s0__neutral__test_iid.summary.json", greedy_pwv)]),
        ("A7 warm start\ngreedy, test-IID", [
            ("warm start", None, "results/eval/a7/a7_init__neutral__test_iid.summary.json", greedy_pwv),
            ("noreport s0", "state", f"{E}/a7_pcstr_d_noreport_s0__neutral__test_iid.summary.json", greedy_pwv),
            ("ack_outcome s0", "ack", f"{E}/a7_ack_outcome_s0__neutral__test_iid.summary.json", greedy_pwv)]),
    ]
    fig, ax = plt.subplots(figsize=(8.0, 4.2))
    for x, (title, pts) in enumerate(panels):
        offs = {None: -0.28, "state": 0.0, "ack": 0.28}
        k = {"state": 0, "ack": 0}
        for label, ch, src, fn in pts:
            if not Path(src).exists():
                continue
            p, n = fn(src)
            data.append({"figure": "F1", "panel": title.replace("\n", ", "), "model": label, "channel": ch or "warm start",
                         "rate": round(p, 4), "n": n, "source": src})
            kw = dict(s=64, zorder=3, linewidths=2, edgecolors=SURF)
            if ch is None:
                ax.scatter(x + offs[ch], p, facecolors="none", edgecolors=INK, s=64, linewidths=1.6, zorder=3)
            else:
                m = sum(c == ch for _, c, _, _ in pts)
                dx = (k[ch] - (m - 1) / 2) * 0.07  # spread a group's arms/seeds so equal rates stay visible
                k[ch] += 1
                ax.scatter(x + offs[ch] + dx, p, color=STATE if ch == "state" else ACK, **kw)
    ax.set_xticks(range(len(panels)), [t for t, _ in panels])
    ax.set_xlim(-0.6, len(panels) - 0.4)
    ax.set_ylim(-0.03, 1.05)
    ax.set_ylabel("share of acked refunds read back")
    ax.yaxis.grid(True, color="#e8e7e3", linewidth=0.8)
    ax.set_axisbelow(True)
    from matplotlib.lines import Line2D
    ax.legend(handles=[Line2D([], [], marker="o", ls="", mfc="none", mec=INK, ms=8, label="warm start (SFT only)"),
                       Line2D([], [], marker="o", ls="", color=STATE, ms=8, label="goal term reads the database"),
                       Line2D([], [], marker="o", ls="", color=ACK, ms=8, label="goal term reads the acknowledgements")],
              loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3, frameon=False, fontsize=8.5)
    fig.tight_layout()
    fig.savefig(OUT / "f1_channel_verification.png", dpi=200)


def f3(data: list) -> None:
    conf = json.load(open("results/confirmatory.json"))
    n = conf["n_tasks"]
    arms = [("rft_v2", "warm start", None), ("tool_trust", "tool_trust", "ack"), ("llm_judge", "LLM judge", "ack"),
            ("outcome", "outcome", "state"), ("pcstr", "PCSTR", "state"), ("pcstr_c", "PCSTR-c", "state"),
            ("pcstr_d", "PCSTR-d", "state")]
    fig, ax = plt.subplots(figsize=(8.0, 3.8))
    for y, (k, name, ch) in enumerate(arms):
        for s, v in enumerate(conf["per_arm_vsc"][k]):
            lo, hi = wilson(v, n)
            data.append({"figure": "F3", "model": name, "seed": s, "vsc": v, "ci95": [round(lo, 4), round(hi, 4)],
                         "n": n, "source": "results/confirmatory.json"})
            yy = y + (s - 0.5) * 0.18 if len(conf["per_arm_vsc"][k]) > 1 else y
            col = INK if ch is None else (STATE if ch == "state" else ACK)
            ax.plot([lo, hi], [yy, yy], color=col, linewidth=2, solid_capstyle="round", alpha=0.45)
            if ch is None:
                ax.scatter(v, yy, facecolors="none", edgecolors=INK, s=56, linewidths=1.6, zorder=3)
            else:
                ax.scatter(v, yy, color=col, s=56, zorder=3, edgecolors=SURF, linewidths=1.5)
    ax.set_yticks(range(len(arms)), [a[1] for a in arms])
    ax.set_xlabel(f"verified-success completion on the confirmatory splits (n = {n})")
    from matplotlib.lines import Line2D
    ax.legend(handles=[Line2D([], [], marker="o", ls="", mfc="none", mec=INK, ms=7, label="warm start"),
                       Line2D([], [], marker="o", ls="-", color=STATE, ms=7, label="state-grounded reward"),
                       Line2D([], [], marker="o", ls="-", color=ACK, ms=7, label="ack / transcript reward")],
              loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3, frameon=False, fontsize=8.5,
              title="dots = training seeds · bars = Wilson 95% CI", title_fontsize=8.5)
    ax.set_xlim(0.55, 0.92)
    ax.xaxis.grid(True, color="#e8e7e3", linewidth=0.8)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(OUT / "f3_before_after.png", dpi=200)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    data: list = []
    f1(data)
    f3(data)
    (OUT / "figures_data.json").write_text(json.dumps(data, indent=1))
    print("\n".join(f"{d['figure']} {d.get('panel', '')} {d['model']}: {d.get('rate', d.get('vsc'))} (n={d['n']})"
                    for d in data))


if __name__ == "__main__":
    main()
