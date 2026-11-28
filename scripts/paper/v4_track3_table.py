"""Track 3 metrics table for main_v4 (descriptive; no new runs).

Reads the committed AD-C1 greedy summaries (results/ad/conf_summaries/*__cw_conf.summary.json) and the batch-1
latency files (results/s2/latency_*.json). Writes paper/tables_v4/track3.tex and results/ad/track3_metrics.json.
Every number is a mean over the seeds of an arm (5 seeds; the warm start is one model).
"""
import json
import pathlib
import statistics as st

ROOT = pathlib.Path(__file__).resolve().parents[2]
SUM = ROOT / "results/ad/conf_summaries"
ARMS = [("Warm start", ["ad_base_warm"]),
        ("GRPO", [f"ad_grpo_s{i}" for i in range(5)]),
        ("Std-free", [f"ad_nostd_s{i}" for i in range(5)]),
        ("AWRB", [f"ad_awrb_s{i}" for i in range(5)]),
        ("AWRB-centred", [f"ad_awrbn_s{i}" for i in range(5)])]


def load(tag):
    return json.loads((SUM / f"{tag}__cw_conf.summary.json").read_text())


def row(name, tags):
    ds = [load(t) for t in tags]
    assert all(d["mode"] == "greedy" and d["split"] == "cw_conf" and d["n"] == 300 for d in ds)
    m = lambda f: st.mean(f(d) for d in ds)
    return {
        "arm": name, "seeds": len(ds),
        "vsc": m(lambda d: d["vsc"]),
        "vsc_min": min(d["vsc"] for d in ds), "vsc_max": max(d["vsc"] for d in ds),
        "vsc_faulted": m(lambda d: d["faulted"]["vsc"]),
        "vsc_clean": m(lambda d: d["by_class"]["cw:clean"]["vsc"]),
        "false_success": m(lambda d: d["false_success"]),
        "tool_call_validity": m(lambda d: d["tool_call_validity"]),
        "calls": m(lambda d: d["calls"]),
        "reads": m(lambda d: d["reads"]),
        "batch_seconds_300": m(lambda d: d["seconds"]),
    }


def main():
    rows = [row(n, t) for n, t in ARMS]
    lat = {t: json.loads((ROOT / f"results/s2/latency_{t}.json").read_text()) for t in ("warm_start", "rv_s0")}
    out = {"source": "results/ad/conf_summaries (AD-C1, greedy, cw_conf, 300 tasks: 179 clean, 121 faulted)",
           "rows": rows,
           "latency_batch1": {t: v["seconds_per_episode"] for t, v in lat.items()},
           "latency_source": "results/s2/latency_*.json (batch 1, proc_dev, 40 episodes, same Tiny Aya Global base)"}
    (ROOT / "results/ad/track3_metrics.json").write_text(json.dumps(out, indent=2) + "\n")

    f = lambda x: f"{x:.3f}"
    lines = [r"\begin{table*}[t]\centering\small",
             r"\begin{tabular}{lcccccccc}\toprule",
             r"Arm (seeds) & \multicolumn{3}{c}{Task success (VSC)} & False & Tool-call & Calls & Reads & Eval s \\",
             r" & all & faulted & clean & success & validity & /ep & /ep & /300 tasks \\\midrule"]
    for r in rows:
        rng = "" if r["seeds"] == 1 else f" {{\\scriptsize [{r['vsc_min']:.3f}, {r['vsc_max']:.3f}]}}"
        lines.append(f"{r['arm']} ({r['seeds']}) & {f(r['vsc'])}{rng} & {f(r['vsc_faulted'])} & {f(r['vsc_clean'])} & "
                     f"{f(r['false_success'])} & {f(r['tool_call_validity'])} & {r['calls']:.2f} & {r['reads']:.2f} & "
                     f"{r['batch_seconds_300']:.0f} \\\\")
    w, v = lat["warm_start"]["seconds_per_episode"], lat["rv_s0"]["seconds_per_episode"]
    lines += [r"\bottomrule\end{tabular}",
              r"\caption{Track 3 metrics for the AD-C1 arms (descriptive; means over seeds, greedy decoding, 300 unseen "
              r"\texttt{cw\_conf} tasks: 121 faulted, 179 clean; range over seeds in brackets). Eval seconds are wall "
              r"time for the whole batched split on one GPU, not per-request latency. Batch-1 latency was measured on "
              r"the same Tiny Aya Global base with a LoRA adapter (40 ProcureSim dev episodes): warm start "
              f"{w['mean']:.2f}\\,s per episode (p95 {w['p95']:.2f}\\,s), database-reward GRPO agent "
              f"{v['mean']:.2f}\\,s (p95 {v['p95']:.2f}\\,s), whose episodes include read-backs. "
              r"Per-run values: \path{results/ad/conf_summaries/}, \path{docs/research/AD_C1_ANALYSIS.md}. Sources: \path{results/ad/track3_metrics.json}, \path{scripts/paper/v4_track3_table.py}.}"
              r"\label{tab:track3}",
              r"\end{table*}"]
    p = ROOT / "paper/tables_v4/track3.tex"
    p.parent.mkdir(exist_ok=True)
    p.write_text("\n".join(lines) + "\n")
    for r in rows:
        print(r["arm"], f(r["vsc"]), f(r["vsc_faulted"]), f(r["vsc_clean"]), f(r["false_success"]),
              f(r["tool_call_validity"]), round(r["reads"], 2))


if __name__ == "__main__":
    main()
