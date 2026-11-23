"""V2-E1 analysis, exactly as registered (docs/research/algorithm_discovery_v2/V2E1_CONDITION_B.md §2-3).
Reads only registered result files; reports missing runs instead of guessing.

    python scripts/ad2/e1_analyze.py --out results/ad2/e1_summary.json
"""
from __future__ import annotations

import argparse
import gzip
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/ad2"))
from decompose import decompose  # noqa: E402

COST = 0.05
RUNS = {  # tag -> (rollout dir, dev dir)
    **{f"ad_{a}_s{s}": ("results/ad/rollouts", "results/ad/dev") for a in ("grpo", "nostd") for s in (0, 1)},
    **{f"ad2_{a}_s{s}": ("results/ad2/rollouts", "results/ad2/dev") for a in ("grpo_rs", "nostd_rs") for s in (0, 1)},
}


def train_curve(tag: str, rdir: str) -> dict | None:
    p = ROOT / rdir / f"{tag}.jsonl.gz"
    if not p.exists():
        return None
    reads, groups = defaultdict(list), defaultdict(list)
    with gzip.open(p, "rt") as f:
        for line in f:
            r = json.loads(line)
            s = int(r["step"] or 0)
            reads[s].append(r["n_verify"])
            groups[(s, r["task_id"])].append(r["score"]["r_v"] - COST * r["n_verify"])
    steps = sorted(reads)
    blk = lambda lo, hi: [x for s in steps if lo <= s < hi for x in reads[s]]
    first, last = blk(0, 20), blk(180, 200)
    zs = [max(v) - min(v) < 1e-12 for (s, _), v in groups.items() if 180 <= s < 200]
    return {"max_step": steps[-1], "reads_first20": sum(first) / max(len(first), 1),
            "reads_last20": sum(last) / len(last) if last else None,
            "zero_std_last20": sum(zs) / len(zs) if zs else None,
            "reads_by_20": {f"{lo}": sum(blk(lo, lo + 20)) / max(len(blk(lo, lo + 20)), 1) for lo in range(0, 200, 20)}}


def dev(tag: str, ddir: str) -> dict | None:
    p = ROOT / ddir / f"{tag}__cw_dev.summary.json"
    if not p.exists():
        return None
    s = json.loads(p.read_text())
    return {"vsc": s["vsc"], "j_c": s["j_c"], "reads": s["reads"], "duplicate": s["duplicate"],
            "faulted_vsc": s["faulted"]["vsc"],
            "verify_after": {k: v[0] for k, v in (s.get("verify_after") or {}).items()}}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    runs = {}
    for tag, (rdir, ddir) in RUNS.items():
        c = train_curve(tag, rdir)
        d = dev(tag, ddir)
        m = None
        if c is not None:
            dd = decompose(tag, 0, 40, COST, rdir)
            m = {"trl_net": dd["trl"]["net"], "none_net": dd["none"]["net"], "groups": dd["groups"]}
        runs[tag] = {"train": c, "dev": d, "decomp_0_40": m}
    P = {}
    g = [runs[f"ad2_grpo_rs_s{s}"] for s in (0, 1)]
    ok = all(x["train"] and x["train"]["reads_last20"] is not None and x["dev"] for x in g)
    if ok:
        P["P1_no_collapse"] = all(x["train"]["reads_last20"] >= 0.5 * x["train"]["reads_first20"] for x in g)
        P["P2_dev_reads_and_jc"] = all(x["dev"]["reads"] > 0.3 and x["dev"]["j_c"] > 0.717 for x in g)
        P["P3_zero_std_below_half"] = all(x["train"]["zero_std_last20"] < 0.5 for x in g)
        P["P4_std_net_above_minus30"] = all(x["decomp_0_40"]["trl_net"] > -30 for x in g)
        P["falsifier_collapse_both_seeds"] = all(x["train"]["reads_last20"] < 0.1 and x["dev"]["reads"] < 0.1 for x in g)
    else:
        P["status"] = "incomplete"
    out = {"runs": runs, "predictions": P, "note": "development, two seeds, descriptive (V2E1_CONDITION_B.md)"}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=1))
    for k, v in runs.items():
        print(k, json.dumps({"train": v["train"] and {x: v["train"][x] for x in ("reads_first20", "reads_last20", "zero_std_last20")},
                             "dev": v["dev"] and {x: v["dev"][x] for x in ("vsc", "j_c", "reads")},
                             "decomp": v["decomp_0_40"] and {x: v["decomp_0_40"][x] for x in ("trl_net", "none_net")}}))
    print(json.dumps(P, indent=1))


if __name__ == "__main__":
    main()
