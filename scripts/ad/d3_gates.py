"""AD-D3 gates and predictions (EXPERIMENT_PROTOCOL_D3.md §4): G1-G4 for nostd_ema_irc vs grpo seeds 0-1; R1-R3 vs
nostd_ema seeds 0-1 (uninformative reads per episode on cw_dev, verify-after-pay, J_c).

    python scripts/ad/d3_gates.py --out results/ad/d3_summary.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from d1_gates import row  # noqa: E402
from d2_gates import gate  # noqa: E402

from actionproof.ad.readinfo import read_flags  # noqa: E402
from actionproof.bench.generator_proc import ProcTask  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]


def uninf_per_ep(tag: str) -> float | None:
    p = ROOT / "results/ad/dev" / f"{tag}__cw_dev.jsonl"
    if not p.exists():
        return None
    n = u = 0
    for line in p.read_text().splitlines():
        r = json.loads(line)
        t = ProcTask.from_json(r["task"])
        n += 1
        u += sum(not ok for _, _, ok in read_flags(t, r["trace"], t.meta["env_seed"]))
    return u / n


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    tags = [f"ad_{e}_s{s}" for e in ("grpo", "nostd_ema", "nostd_ema_irc") for s in (0, 1)]
    runs = {t: row(t) for t in tags}
    for t in tags:
        if runs[t] is not None:
            runs[t]["uninf_per_ep"] = uninf_per_ep(t)
    missing = [k for k, v in runs.items() if v is None]
    g = [runs[f"ad_grpo_s{s}"] for s in (0, 1)]
    c = [runs[f"ad_nostd_ema_irc_s{s}"] for s in (0, 1)]
    b = [runs[f"ad_nostd_ema_s{s}"] for s in (0, 1)]
    out = {"runs": runs, "missing": missing}
    if None not in g and None not in c:
        out["gates"] = gate(c, g)
    if None not in b and None not in c:
        m = lambda k, xs: sum(x[k] for x in xs) / len(xs)
        ub, uc = m("uninf_per_ep", b), m("uninf_per_ep", c)
        vb = sum(x["verify_after"]["pay"] or 0 for x in b) / 2
        vc = sum(x["verify_after"]["pay"] or 0 for x in c) / 2
        out["predictions"] = {"R1_uninf_drop_share": (ub - uc) / ub if ub else None, "R1": ub > 0 and (ub - uc) / ub >= 0.5,
                              "R2_va_pay_diff": vc - vb, "R2": vc - vb >= -0.05,
                              "R3_j_c_diff": m("j_c", c) - m("j_c", b), "R3": m("j_c", c) >= m("j_c", b)}
    Path(a.out).write_text(json.dumps(out, indent=1))
    print(json.dumps({k: v for k, v in out.items() if k != "runs"}, indent=1))


if __name__ == "__main__":
    main()
