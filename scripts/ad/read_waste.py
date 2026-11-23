"""Share of reads that are provably uninformative under the cw1 prior (actionproof/ad/readinfo.py), per model, on a
CW evaluation file. Descriptive. The J_c lost to them is c times the uninformative reads per episode.

    python scripts/ad/read_waste.py --dir results/ad/dev --split cw_dev --tags ad_grpo_s0 ad_nostd_s0 ...
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from actionproof.ad.readinfo import read_flags
from actionproof.bench.generator_proc import ProcTask

ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="results/ad/dev")
    ap.add_argument("--split", default="cw_dev")
    ap.add_argument("--tags", nargs="+", required=True)
    ap.add_argument("--cost", type=float, default=0.05)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    res = {}
    for tag in a.tags:
        p = ROOT / a.dir / f"{tag}__{a.split}.jsonl"
        if not p.exists():
            continue
        n_ep = inf = uninf = 0
        by_tool: Counter = Counter()
        for line in p.read_text().splitlines():
            r = json.loads(line)
            t = ProcTask.from_json(r["task"])
            n_ep += 1
            for _, tool, ok in read_flags(t, r["trace"], t.meta["env_seed"]):
                inf += ok
                uninf += not ok
                if not ok:
                    by_tool[tool] += 1
        res[tag] = {"reads_per_ep": (inf + uninf) / n_ep, "uninformative_per_ep": uninf / n_ep,
                    "uninformative_share": uninf / max(1, inf + uninf), "j_c_lost": a.cost * uninf / n_ep,
                    "uninformative_by_tool": dict(by_tool)}
        print(tag, json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in res[tag].items()}))
    if a.out:
        Path(a.out).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
