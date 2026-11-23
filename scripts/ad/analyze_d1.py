"""AD-D1 learning curves and behavioural mechanism (descriptive; reads registered outputs only).

Training (results/ad/rollouts/<tag>.jsonl.gz, one line per episode with its step), per block of 20 steps:
  reads per episode, verify-after rate per write site, VSC, R_V, E[R_V | traj], duplicate rate, and the reward-signal
  noise: within-group std of the raw return R_c vs of the Rao-Blackwellised return.
Evaluation (results/ad/dev/<tag>__cw_dev.jsonl, greedy): per site, P(read | ok ack) split by whether the write
actually failed (aliased, so a policy cannot differ: a sanity check), P(repair | read revealed a fault), reads after
the journal entry (zero value under cw1), PO reads, duplicates.

    python scripts/ad/analyze_d1.py --out results/ad/d1_mechanism.json [--plots docs/research/algorithm_discovery/figures]
"""
from __future__ import annotations

import argparse
import gzip
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev

from actionproof.ad.evaluate import verify_after

ROOT = Path(__file__).resolve().parents[2]
COST = {"": 0.05, "c30": 0.30}
TAGS = [f"ad_{e}_s{s}" for e in ("grpo", "nostd", "awrb", "awrbn") for s in (0, 1)] + ["ad_awrbc30_s0", "ad_awrbnc30_s0"]


def cost_of(tag: str) -> float:
    return 0.30 if "c30" in tag else 0.05


def training_curves(tag: str, block: int = 20) -> dict | None:
    p = ROOT / "results/ad/rollouts" / f"{tag}.jsonl.gz"
    if not p.exists():
        return None
    c = cost_of(tag)
    by_step: dict[int, list[dict]] = defaultdict(list)
    with gzip.open(p, "rt") as f:
        for line in f:
            r = json.loads(line)
            by_step[int(r["step"] or 0)].append(r)
    blocks = []
    steps = sorted(by_step)
    for b0 in range(0, (max(steps) if steps else 0) + 1, block):
        rs = [r for s in steps if b0 <= s < b0 + block for r in by_step[s]]
        if not rs:
            continue
        va = defaultdict(lambda: [0, 0])
        for r in rs:
            for k, (x, y) in verify_after(r).items():
                va[k][0] += x
                va[k][1] += y
        noise_raw, noise_rb = [], []
        for s in steps:
            if not b0 <= s < b0 + block:
                continue
            groups = defaultdict(list)
            for r in by_step[s]:
                groups[r["task_id"]].append(r)
            for g in groups.values():
                if len(g) > 1:
                    noise_raw.append(pstdev([x["score"]["r_v"] - c * x["n_verify"] for x in g]))
                    noise_rb.append(pstdev([(x["ad"]["r_bar"] or 0) - c * x["n_verify"] for x in g]))
        blocks.append({"step": b0, "n": len(rs), "reads": mean(r["n_verify"] for r in rs),
                       "vsc": mean(r["score"]["vsc"] for r in rs), "r_v": mean(r["score"]["r_v"] for r in rs),
                       "r_bar": mean(r["ad"]["r_bar"] or 0 for r in rs),
                       "dup": mean(r["score"]["duplicate_irreversible"] for r in rs),
                       "verify_after": {k: x / y for k, (x, y) in va.items() if y},
                       "group_std_raw": mean(noise_raw) if noise_raw else None,
                       "group_std_rb": mean(noise_rb) if noise_rb else None,
                       "zero_std_raw": mean(x < 1e-9 for x in noise_raw) if noise_raw else None,
                       "zero_std_rb": mean(x < 1e-9 for x in noise_rb) if noise_rb else None})
    return {"blocks": blocks}


def eval_mechanism(tag: str) -> dict | None:
    p = ROOT / "results/ad/dev" / f"{tag}__cw_dev.jsonl"
    if not p.exists():
        return None
    rs = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
    site = defaultdict(lambda: {"read_given_faulted": [0, 0], "read_given_clean": [0, 0],
                                "repair_given_revealed": [0, 0]})
    for r in rs:
        fault = (r["task"]["meta"].get("faults") or [None])[0]
        va = verify_after(r)
        for step, (x, y) in va.items():
            faulted = fault is not None and fault["tool"].startswith({"order": "place", "pay": "pay",
                                                                         "receive": "record", "post": "post"}[step])
            key = "read_given_faulted" if faulted else "read_given_clean"
            site[step][key][0] += x
            site[step][key][1] += y
            if faulted and x:
                site[step]["repair_given_revealed"][0] += r["score"]["steps"].get(step) == "VERIFIED_SUCCESS"
                site[step]["repair_given_revealed"][1] += 1
    out = {s: {k: (a / b if b else None, b) for k, (a, b) in v.items()} for s, v in site.items()}
    return {"by_site": out, "post_reads_per_ep": mean(sum(x["tool"] == "get_journal_entries" for x in r["trace"]) for r in rs),
            "po_reads_per_ep": mean(sum(x["tool"] == "get_purchase_order" for x in r["trace"]) for r in rs),
            "duplicate": mean(r["score"]["duplicate_irreversible"] for r in rs)}


def plots(res: dict, outdir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    outdir.mkdir(parents=True, exist_ok=True)
    colors = {"grpo": "#4C72B0", "nostd": "#DD8452", "awrb": "#55A868", "awrbn": "#C44E52"}
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
    for tag, r in res.items():
        tc = r.get("train")
        if not tc or "c30" in tag:
            continue
        est = tag.split("_")[1]
        xs = [b["step"] for b in tc["blocks"]]
        ls = "-" if tag.endswith("s0") else "--"
        axes[0].plot(xs, [b["reads"] for b in tc["blocks"]], ls, color=colors[est], label=tag)
        axes[1].plot(xs, [b["verify_after"].get("pay", float("nan")) for b in tc["blocks"]], ls, color=colors[est])
        axes[2].plot(xs, [b["r_v"] for b in tc["blocks"]], ls, color=colors[est])
    for ax, t in zip(axes, ("reads per episode", "verify after payment", "R_V (training, T=1)")):
        ax.set_title(t)
        ax.set_xlabel("training step")
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(outdir / "ad_d1_training_curves.png", dpi=130)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--plots", default=None)
    a = ap.parse_args()
    res = {}
    for t in TAGS + ["ad_base_warm", "ad_base_a12rv_s0", "ad_base_rsc_s0"]:
        res[t] = {"train": training_curves(t) if not t.startswith("ad_base") else None, "eval": eval_mechanism(t)}
    Path(a.out).write_text(json.dumps(res, indent=1))
    for t, r in res.items():
        if r["train"]:
            b = r["train"]["blocks"]
            print(t, "train first/last block reads", round(b[0]["reads"], 2), round(b[-1]["reads"], 2),
                  "va_pay", {k: round(v, 2) for k, v in b[-1]["verify_after"].items()},
                  "std raw/rb", b[-1]["group_std_raw"] and round(b[-1]["group_std_raw"], 3),
                  b[-1]["group_std_rb"] and round(b[-1]["group_std_rb"], 3))
        if r["eval"]:
            print(t, "eval", json.dumps(r["eval"])[:400])
    if a.plots:
        plots(res, ROOT / a.plots)


if __name__ == "__main__":
    main()
