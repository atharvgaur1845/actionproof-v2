"""Cross-seed ranges of the A12 reward-signal analysis (exploratory), for reporting: per arm, over every seed file
results/a12/reward_signal_s<seed>.json and every 50-step block, the min and max of
  phantom informative share, phantom read gap, lost-ack + never-commit informative share, and their read gaps;
plus the read rate in the first and last block.
    python scripts/a12_reward_signal_summary.py --out results/s2/reward_signal_summary.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def rng(vals: list) -> list | None:
    v = [x for x in vals if x is not None]
    return [round(min(v), 4), round(max(v), 4)] if v else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/s2/reward_signal_summary.json")
    a = ap.parse_args()
    files = sorted((ROOT / "results/a12").glob("reward_signal_s*.json"))
    res: dict = {"seeds": [f.stem.split("_s")[-1] for f in files], "note": "exploratory; ranges over seeds and 50-step blocks"}
    for arm in ("proc_rv", "proc_ra"):
        ph_share, ph_gap, la_nc_share, la_nc_gap, first_rr, last_rr = [], [], [], [], [], []
        for f in files:
            d = json.loads(f.read_text())[arm]
            blocks = sorted(d, key=lambda b: int(b.split("-")[0]))
            for b in blocks:
                c = d[b]
                ph_share.append(c.get("phantom", {}).get("informative_share"))
                ph_gap.append(c.get("phantom", {}).get("read_gap"))
                la_nc_share.append(sum((c.get(k, {}).get("informative_share") or 0) for k in ("lost_ack", "never_commit")))
                la_nc_gap += [c.get(k, {}).get("read_gap") for k in ("lost_ack", "never_commit")]

            def rate(b: str) -> float:
                rows = d[b].values()
                return sum(r["read_rate"] * r["groups"] for r in rows) / sum(r["groups"] for r in rows)
            first_rr.append(rate(blocks[0]))
            last_rr.append(rate(blocks[-1]))
        res[arm] = {"phantom_informative_share": rng(ph_share), "phantom_read_gap": rng(ph_gap),
                    "lost_ack_never_commit_informative_share": rng(la_nc_share),
                    "lost_ack_never_commit_read_gap": rng(la_nc_gap),
                    "read_rate_first_block": rng(first_rr), "read_rate_last_block": rng(last_rr)}
    out = ROOT / a.out
    out.write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
