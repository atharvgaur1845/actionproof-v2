"""Track 3 trajectory metrics (task success, tool-call accuracy, evaluation wall time) from stored episodes.

Descriptive, not registered. Reads only existing evaluation files; runs no model.

Definitions (per episode record, the schema written by scripts/p1_eval.py `run`, reused by p4_eval.py and
txn_eval.py, and by actionproof/ad/evaluate.py `run`):
  task success     P1, P4, AD-C1: `vsc` (verified success: every requested step VERIFIED_SUCCESS in the database and
                   a correct final report). TXN: `txn.vtc` (verified transaction completion, the TXN paper metric).
                   Greedy (temperature 0) and sampled (temperature 1) episodes are reported separately.
  policy calls     the policy's own tool calls: trace entries after the scripted start prefix (`policy_calls`
                   when the record has it, else the agent-made trace entries after `task.meta.start_prefix`), plus
                   calls whose arguments were not valid JSON (they never reach the environment, so they are not in
                   the trace; they are counted in `invalid`).
  tool-call acc.   1 - invalid / policy calls, pooled over episodes. `invalid` counts calls that named an unknown
                   tool, had arguments that do not bind to the tool signature, or had unparseable JSON arguments.
                   A turn with no parseable tool call ends the episode (`ended_by == "no_call"`); it is not a call,
                   and its rate is reported separately.
  calls / episode  policy calls per episode (as above).  turns / episode: model turns (`turns`).
  wall s / episode evaluation `seconds` in the run's summary file / episodes in that run. The run is one batched
                   vLLM job over all of its episodes (P1, P4, TXN: greedy and sampled together; AD-C1: one file per
                   decoding mode), so this is throughput time per episode, not per-request latency. The GPU is the
                   node in queue/claims/<task>/owner of the task that wrote the file (provenance file).
No result file records per-request or per-episode latency.

    # on the cluster, once: GPU provenance from the task queue (read-only)
    python scripts/t3_metrics.py provenance --root /scratch/pabitra/actionproof --queue /scratch/pabitra/actionproof/queue
    # tables, claims block
    python scripts/t3_metrics.py build --root /scratch/pabitra/actionproof
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
OUT = Path("results/t3_metrics")
BAD = ("unknown_tool", "invalid_arguments")
GPU = {"gpunode7": "H200", "gpunode4": "A100 80GB", "gpunode5": "H100"}
P1_SPLITS = ["p1_test_full", "p1_test_pp", "p1_test_rec", "p1_test_wf4", "p1_test_combo"]
P4_SPLITS = ["s4_api", "s4_wf4"]
AD_MODES = ["", ".t1_s0", ".t1_s1", ".t1_s2", ".t1_s3"]

# (study, arm label, [(tag, task id that wrote it)], shown in the table)
P1_ARMS = [("Dw (warm start)", [("Dw", "1620_p1_dwarm_eval")], True),
           ("A (RSC GRPO)", [("A_s0", "1160_p1_test_A_s0"), ("A_s1", "1164_p1_test_A_s1"),
                             ("A_s2", "1168_p1_test_A_s2")], True),
           ("C (A + recovery SFT)", [("C_s0", "1162_p1_test_C_s0"), ("C_s1", "1166_p1_test_C_s1"),
                                     ("C_s2", "1170_p1_test_C_s2")], False),
           ("D (Dw + GRPO)", [("D_s0", "1163_p1_test_D_s0"), ("D_s1", "1167_p1_test_D_s1"),
                              ("D_s2", "1171_p1_test_D_s2")], True),
           ("Dn (Dw + std-free GRPO)", [("Dn_s0", "1610_p3_dn_s0"), ("Dn_s1", "1611_p3_dn_s1"),
                                        ("Dn_s2", "1612_p3_dn_s2")], False)]
P4_ARMS = [("Dw (warm start)", [("Dw_s0", "1716_p4_Dw_s0"), ("Dw_s1", "1714_p4_Dw_s1"),
                                ("Dw_s2", "1715_p4_Dw_s2")], True),
           ("Dw_aug (augmented warm start)", [("Dwaug_s0", "1711_p4_Dwaug_s0"), ("Dwaug_s1", "1712_p4_Dwaug_s1"),
                                              ("Dwaug_s2", "1713_p4_Dwaug_s2")], True)]
TXN_ARMS = [("B0 (script: trust the reply)", [("B0", "1704_txn_policies")], True),
            ("B4 (script: reconciliation)", [("B4", "1704_txn_policies")], True),
            ("Dw (warm start)", [("Dw", "1706_txn_trained_a")], True),
            ("A (RSC GRPO)", [("A_s0", "1706_txn_trained_a"), ("A_s1", "1706_txn_trained_a"),
                              ("A_s2", "1706_txn_trained_a")], False),
            ("D (Dw + GRPO)", [("D_s0", "1707_txn_trained_d"), ("D_s1", "1707_txn_trained_d"),
                               ("D_s2", "1707_txn_trained_d")], True),
            ("Dn (Dw + std-free GRPO)", [("Dn_s0", "1707_txn_trained_d"), ("Dn_s1", "1707_txn_trained_d"),
                                         ("Dn_s2", "1707_txn_trained_d")], False),
            ("Dw_aug (augmented warm start)", [("Dwaug_s0", "1718_txn_dwaug"), ("Dwaug_s1", "1718_txn_dwaug"),
                                               ("Dwaug_s2", "1718_txn_dwaug")], True)]
AD_ARMS = [("warm start", [("ad_base_warm", "1241_ad_c1_eval_b")], True),
           ("GRPO", [(f"ad_grpo_s{k}", "1240_ad_c1_eval_a") for k in range(5)], True),
           ("std-free", [(f"ad_nostd_s{k}", "1241_ad_c1_eval_b") for k in range(5)], True)]


def groups() -> list[dict[str, Any]]:
    """Every (study, subset, arm) cell with its episode files (each paired with its summary and task)."""
    out = []
    for split in P1_SPLITS:
        for arm, tags, show in P1_ARMS:
            out.append({"study": "P1 test", "subset": split, "arm": arm, "show": show, "metric": "vsc",
                        "runs": [(f"results/p1/test/{t}__{split}.jsonl", task) for t, task in tags]})
    for split in P4_SPLITS:
        for arm, tags, show in P4_ARMS:
            out.append({"study": "P4", "subset": split, "arm": arm, "show": show, "metric": "vsc",
                        "runs": [(f"results/p4/{t}__{split}.jsonl", task) for t, task in tags]})
    for arm, tags, show in TXN_ARMS:
        out.append({"study": "TXN", "subset": "txn_test", "arm": arm, "show": show, "metric": "vtc",
                    "runs": [(f"results/txn/{t}__txn_test.jsonl", task) for t, task in tags]})
    for arm, tags, show in AD_ARMS:
        out.append({"study": "AD-C1", "subset": "cw_conf", "arm": arm, "show": show, "metric": "vsc",
                    "runs": [(f"results/ad/conf/{t}__cw_conf{m}.jsonl", task) for t, task in tags for m in AD_MODES]})
    return out


def summary_path(jsonl: str) -> str:
    return jsonl[: -len(".jsonl")] + ".summary.json"


# ----------------------------------------------------------------------------------------------- per episode
def episode(r: dict[str, Any], metric: str) -> dict[str, Any]:
    trace = r.get("trace") or []
    if r.get("policy_calls") is not None:
        n0 = len(trace) - int(r["policy_calls"])
    else:
        n0 = len(((r.get("task") or {}).get("meta") or {}).get("start_prefix") or [])
    agent = [c for c in trace[n0:] if c.get("by", "agent") == "agent"]
    env_bad = sum((c.get("obs") or {}).get("error") in BAD for c in agent)
    invalid = int(r["invalid"]) if r.get("invalid") is not None else env_bad
    unparsed = invalid - env_bad
    if unparsed < 0:
        raise ValueError(f"{r.get('rollout_id')}: invalid {invalid} < invalid calls in trace {env_bad}")
    if r.get("mode") is not None:
        greedy = r["mode"] == "greedy"
    else:
        greedy = r.get("sample", -1) < 0
    success = (r.get("txn") or {}).get("vtc") if metric == "vtc" else r.get("vsc")
    return {"greedy": greedy, "success": bool(success), "calls": len(agent) + unparsed, "invalid": invalid,
            "turns": r.get("turns"), "no_call": r.get("ended_by") == "no_call"}


def aggregate(eps: list[dict[str, Any]]) -> dict[str, Any] | None:
    n = len(eps)
    if not n:
        return None
    calls = sum(e["calls"] for e in eps)
    turns = [e["turns"] for e in eps if e["turns"] is not None]
    return {"episodes": n, "success": sum(e["success"] for e in eps) / n,
            "policy_calls": calls, "invalid_calls": sum(e["invalid"] for e in eps),
            "tool_call_accuracy": (1 - sum(e["invalid"] for e in eps) / calls) if calls else None,
            "calls_per_episode": calls / n,
            "turns_per_episode": (sum(turns) / len(turns)) if len(turns) == n else None,
            "no_call_end_rate": sum(e["no_call"] for e in eps) / n}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def cell(g: dict[str, Any], root: Path, prov: dict[str, Any]) -> dict[str, Any]:
    eps, runs = [], []
    for rel, task in g["runs"]:
        rows = read_jsonl(root / rel)
        eps += [episode(r, g["metric"]) for r in rows]
        summ = json.loads((root / summary_path(rel)).read_text())
        secs = summ.get("seconds")
        p = prov.get(summary_path(rel)) or {}
        runs.append({"file": rel, "summary": summary_path(rel), "task": task, "episodes": len(rows),
                     "seconds": secs, "wall_s_per_episode": (secs / len(rows)) if secs is not None else None,
                     "node": p.get("node"), "gpu": p.get("gpu") if p.get("consistent") else None})
    lat = [r["wall_s_per_episode"] for r in runs if r["wall_s_per_episode"] is not None]
    gpus = sorted({r["gpu"] or "unattributed" for r in runs if r["wall_s_per_episode"] is not None})
    return {"study": g["study"], "subset": g["subset"], "arm": g["arm"], "in_table": g["show"],
            "success_metric": g["metric"], "n_models": len({r["file"].split("/")[-1].split("__")[0] for r in runs}),
            "greedy": aggregate([e for e in eps if e["greedy"]]), "sampled": aggregate([e for e in eps if not e["greedy"]]),
            "all": aggregate(eps),
            "wall_s_per_episode": {"min": min(lat), "max": max(lat), "pooled": sum(r["seconds"] for r in runs
                                   if r["seconds"] is not None) / sum(r["episodes"] for r in runs if r["seconds"] is not None),
                                   "gpus": gpus} if lat else None,
            "runs": runs}


def key_of(c: dict[str, Any]) -> str:
    arm = c["arm"].split(" (")[0]
    return f"{c['study'].split()[0]}|{c['subset']}|{arm}"


# ----------------------------------------------------------------------------------------------- formatting
def f3(x: float | None) -> str:
    return "--" if x is None else f"{x:.3f}"


def f1(x: float | None) -> str:
    return "--" if x is None else f"{x:.1f}"


def f2(x: float | None) -> str:
    return "--" if x is None else f"{x:.2f}"


def lat_text(w: dict[str, Any] | None, sep: str = "--") -> tuple[str, str]:
    if not w:
        return "--", "--"
    v = f2(w["min"]) if f2(w["min"]) == f2(w["max"]) else f"{f2(w['min'])}{sep}{f2(w['max'])}"
    gpu_short = {"A100 80GB": "A100", "H100": "H100", "H200": "H200", "unattributed": "?"}
    return v, "/".join(gpu_short[g] for g in w["gpus"])


def row_values(c: dict[str, Any], sep: str = "--") -> list[str]:
    a, g, s = c["all"], c["greedy"], c["sampled"]
    lat, gpu = lat_text(c["wall_s_per_episode"], sep)
    return [f3(g["success"]) if g else "--", f3(s["success"]) if s else "--", f3(a["tool_call_accuracy"]),
            f1(a["calls_per_episode"]), f1(a["turns_per_episode"]), lat, gpu]


SUBSET_TEX = {"p1_test_full": "full", "p1_test_pp": "partial pay", "p1_test_rec": "recovery start",
              "p1_test_wf4": "4-step workflow", "p1_test_combo": "combined faults", "s4_api": "renamed tools",
              "s4_wf4": "unseen 4-step", "txn_test": "txn\\_test", "cw_conf": "cw\\_conf"}
CAPTION = (
    "Track~3 trajectory metrics, descriptive, not registered (\\texttt{scripts/t3\\_metrics.py}, from stored "
    "evaluation episodes only; no new runs). Rows pool the arm's seeds (models column). \\emph{Success}: verified "
    "success VSC (P1, P4, AD-C1) or verified transaction completion VTC (TXN), greedy and sampled ($T=1$) episodes "
    "separately. \\emph{Tool acc.}: $1-$invalid$/$policy calls, pooled over all episodes; invalid = unknown tool, "
    "arguments that do not bind to the tool signature, or arguments that are not valid JSON; policy calls exclude "
    "the scripted start prefix. A turn without a parseable call ends the episode and is not a call. \\emph{Calls}, "
    "\\emph{turns}: per episode, all episodes. \\emph{Wall s/ep}: the run's evaluation wall time divided by its "
    "episodes (range over runs); each run is one batched vLLM job, so this is throughput time per episode on the "
    "named GPU, not per-request latency, and it depends on the GPU, batch size and output length. No file records "
    "per-request latency. Scripted policies (B0, B4) have no model turns or timed runs.")


def tex(cells: list[dict[str, Any]]) -> str:
    lines = ["% Generated by scripts/t3_metrics.py from results/t3_metrics/t3_metrics.json; do not edit by hand.",
             "\\begin{table*}[t]", "\\centering", "\\small", "\\setlength{\\tabcolsep}{4pt}",
             "\\begin{tabular}{lllrrrrrrl}", "\\toprule",
             "Study & Subset & Arm & Models & Success (greedy) & Success (sampled) & Tool acc. & Calls & Turns & "
             "Wall s/ep (GPU) \\\\", "\\midrule"]
    prev = None
    for c in cells:
        if not c["in_table"]:
            continue
        if prev and prev != c["study"]:
            lines.append("\\midrule")
        prev = c["study"]
        v = row_values(c)
        arm = c["arm"].replace("_", "\\_")
        wall = "--" if v[5] == "--" else f"{v[5]} ({v[6]})"
        lines.append(f"{c['study']} & {SUBSET_TEX[c['subset']]} & {arm} & {c['n_models']} & {v[0]} & {v[1]} & "
                     f"{v[2]} & {v[3]} & {v[4]} & {wall} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", f"\\caption{{{CAPTION}}}", "\\label{tab:t3metrics}",
              "\\end{table*}", ""]
    return "\n".join(lines)


def markdown(cells: list[dict[str, Any]], sources: list[str]) -> str:
    out = ["# Track 3 trajectory metrics (descriptive, not registered)", "",
           "Generated by `scripts/t3_metrics.py` from stored evaluation episodes only (no new model runs). The LaTeX "
           "version is `paper/t3_metrics_table.tex`; all values, per-run times and per-mode breakdowns are in "
           "`results/t3_metrics/t3_metrics.json`.", "",
           "## Definitions", "",
           "- **Task success**: verified success VSC (P1, P4, AD-C1): every requested step is verified in the "
           "database and the final report is correct. TXN uses verified transaction completion VTC (`txn.vtc`). "
           "Greedy (temperature 0) and sampled (temperature 1) episodes are reported separately.",
           "- **Tool-call accuracy**: 1 minus invalid calls over policy calls, pooled over all episodes of the row. "
           "Invalid (`invalid` in each episode record) counts calls naming an unknown tool, calls whose arguments do "
           "not bind to the tool signature, and calls whose arguments are not valid JSON. Policy calls are the "
           "policy's own trace entries after the scripted start prefix (`policy_calls`), plus the invalid-JSON calls, "
           "which never reach the environment. A turn with no parseable tool call ends the episode; it is not a "
           "call (its rate is `no_call_end_rate` in the JSON).",
           "- **Calls, turns**: policy calls and model turns per episode, over all episodes of the row.",
           "- **Wall s/ep**: evaluation wall time (`seconds` in the run's summary file) divided by the episodes of "
           "that run; the range is over the row's runs. Each run is one batched vLLM job (P1, P4, TXN: greedy and "
           "sampled episodes together; AD-C1: one job per decoding mode), so this is throughput time per episode on "
           "the named GPU, not per-request latency. It depends on GPU, batch size and output length, so rows on "
           "different GPUs are not comparable. The GPU comes from `queue/claims/<task>/owner` of the task that wrote "
           "the summary (gpunode7 H200, gpunode4 A100 80GB, gpunode5 H100), kept only when the summary was written "
           "after that task's claim time (`results/t3_metrics/gpu_provenance.json`).",
           "- **Not supported by the files**: per-request or per-episode latency (no file records it). Scripted "
           "policies B0 and B4 have no model turns and no timed run. The Stage 2 file `results/latency_single.json` "
           "measured a different, older model and is not used.", "",
           "## Table", "",
           "| Study | Subset | Arm | Models | Success (greedy) | Success (sampled) | Tool acc. | Calls/ep | Turns/ep "
           "| Wall s/ep | GPU |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    for c in cells:
        if not c["in_table"]:
            continue
        v = row_values(c, " to ")
        out.append(f"| {c['study']} | `{c['subset']}` | {c['arm']} | {c['n_models']} | " + " | ".join(v) + " |")
    out += ["", "Arms C and Dn (P1, TXN) and A (TXN) are in the JSON but not in the table.", "",
            "## Regenerate", "", "On sharanga, from a checkout of this branch (`--root` is only read; outputs, "
            "including the T3M block of `claims/claims.yaml`, are written into the checkout):", "",
            "```", "python scripts/t3_metrics.py provenance --root /scratch/pabitra/actionproof "
            "--queue /scratch/pabitra/actionproof/queue",
            "python scripts/t3_metrics.py build --root /scratch/pabitra/actionproof", "```", "",
            "## Source files", "", "Episode files (`.jsonl`, not in git; on sharanga under "
            "`/scratch/pabitra/actionproof`) and their `.summary.json` files:", ""]
    out += [f"- `{s}`" for s in sources]
    out.append("")
    return "\n".join(out)


def claims_fragment(cells: list[dict[str, Any]]) -> str:
    """The T3M claim block for claims/claims.yaml: every number in the table, tied to t3_metrics.json."""
    f = "results/t3_metrics/t3_metrics.json"
    lines = ["  # >>> T3M (generated by scripts/t3_metrics.py; do not edit by hand)", "  - id: T3M",
             '    statement: "Track 3 trajectory metrics (descriptive, not registered): task success (VSC or VTC), '
             'tool-call accuracy, policy calls and turns per episode, and batched evaluation wall time per episode, '
             'recomputed by scripts/t3_metrics.py from stored evaluation episodes."',
             "    status: supported", "    eval: [results/t3_metrics/t3_metrics.json, "
             "results/t3_metrics/gpu_provenance.json]", "    stats: scripts/t3_metrics.py", "    numbers:"]
    for i, c in enumerate(cells):
        if not c["in_table"]:
            continue
        k = key_of(c)
        items = []
        for mode in ("greedy", "sampled"):
            if c[mode]:
                items.append((f"{mode}_success", [k, mode, "success"], f3(c[mode]["success"])))
        items += [("tool_acc", [k, "all", "tool_call_accuracy"], f3(c["all"]["tool_call_accuracy"])),
                  ("calls", [k, "all", "calls_per_episode"], f1(c["all"]["calls_per_episode"]))]
        if c["all"]["turns_per_episode"] is not None:
            items.append(("turns", [k, "all", "turns_per_episode"], f1(c["all"]["turns_per_episode"])))
        if c["wall_s_per_episode"]:
            items += [("wall_min", [k, "wall_s_per_episode", "min"], f2(c["wall_s_per_episode"]["min"])),
                      ("wall_max", [k, "wall_s_per_episode", "max"], f2(c["wall_s_per_episode"]["max"]))]
        for name, key, val in items:
            lines.append(f"      r{i:02d}_{name}: {{value: {val}, file: {f}, key: {json.dumps(['cells'] + key)}, "
                         f"shown: [\"{val}\"]}}")
    lines.append("  # <<< T3M")
    return "\n".join(lines) + "\n"


def sync_claims(fragment: str, path: Path) -> None:
    text = path.read_text()
    start, end = "  # >>> T3M", "  # <<< T3M\n"
    if start in text:
        a, b = text.index(start), text.index(end) + len(end)
        text = text[:a] + fragment + text[b:]
    else:
        a = text.index("\nallow:") + 1
        text = text[:a] + fragment + "\n" + text[a:]
    path.write_text(text)


# ----------------------------------------------------------------------------------------------- commands
def build(root: Path, out_root: Path) -> dict[str, Any]:
    pf = out_root / OUT / "gpu_provenance.json"
    prov = json.loads(pf.read_text()) if pf.exists() else {}
    cells = [cell(g, root, prov) for g in groups()]
    sources = sorted({p for c in cells for r in c["runs"] for p in (r["file"], r["summary"])})
    data = {"description": "Track 3 trajectory metrics, descriptive, not registered; see scripts/t3_metrics.py "
                           "for definitions", "cells": {key_of(c): c for c in cells}, "sources": sources}
    (out_root / OUT).mkdir(parents=True, exist_ok=True)
    (out_root / OUT / "t3_metrics.json").write_text(json.dumps(data, indent=1, sort_keys=True) + "\n")
    (out_root / "paper").mkdir(exist_ok=True)
    (out_root / "paper/t3_metrics_table.tex").write_text(tex(cells))
    (out_root / "docs/submission").mkdir(parents=True, exist_ok=True)
    (out_root / "docs/submission/T3_METRICS.md").write_text(markdown(cells, sources))
    frag = claims_fragment(cells)
    (out_root / OUT / "t3_claims.yaml").write_text(frag)
    if (out_root / "claims/claims.yaml").exists():
        sync_claims(frag, out_root / "claims/claims.yaml")
    return data


def provenance(root: Path, queue: Path, out_root: Path) -> dict[str, Any]:
    """Node and GPU of the task that wrote each summary file, kept only if the file was written after the claim."""
    out = {}
    for g in groups():
        for rel, task in g["runs"]:
            s = summary_path(rel)
            owner = (queue / "claims" / task / "owner").read_text().strip()
            _, job, host, when = owner.split(":", 3)
            node = host.split(".")[0]
            claimed = dt.datetime.fromisoformat(when)
            mtime = dt.datetime.fromtimestamp(os.stat(root / s).st_mtime, dt.timezone.utc)
            out[s] = {"task": task, "owner": owner, "node": node, "gpu": GPU.get(node),
                      "claimed": claimed.isoformat(), "summary_mtime_utc": mtime.isoformat(timespec="seconds"),
                      "consistent": mtime >= claimed}
    (out_root / OUT).mkdir(parents=True, exist_ok=True)
    (out_root / OUT / "gpu_provenance.json").write_text(json.dumps(out, indent=1, sort_keys=True) + "\n")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("build", "provenance"))
    ap.add_argument("--root", type=Path, default=ROOT, help="directory holding results/ (read only)")
    ap.add_argument("--out-root", type=Path, default=ROOT, help="repository root to write outputs into")
    ap.add_argument("--queue", type=Path, default=None)
    a = ap.parse_args()
    if a.cmd == "provenance":
        p = provenance(a.root, a.queue, a.out_root)
        print(f"{len(p)} summaries, {sum(not v['consistent'] for v in p.values())} not attributable")
    else:
        d = build(a.root, a.out_root)
        print(f"{len(d['cells'])} cells from {len(d['sources'])} files")
    return 0


if __name__ == "__main__":
    sys.exit(main())
