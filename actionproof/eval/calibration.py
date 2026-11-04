"""Calibrated trust (H8, upgrade 1): does an agent verify unreliable services more often?

For each service s: check_rate(s) = P(a success-acked write to s is followed by a read of the SAME
entity before the episode ends). Calibration = Spearman rho across services between check_rate and
the hidden fault rate rho_s of the split (training priors, or the swapped priors for
test_calib_swap), with a task-level bootstrap CI. A uniform verify-everything prompt has a flat
profile (rho ~ undefined/0); a calibrated agent's profile follows rho_s.

    python -m actionproof.eval.calibration results/eval/r3/grpo_tiny_v2_pcstr_c_s0__neutral__test_iid.jsonl
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict

from actionproof.bench.generator import SWAPPED_RHO, TRAIN_RHO
from actionproof.env.tools_ecom import TOOL_META, VERIFICATION_TOOLS
from actionproof.eval.by_fault import ID_ARGS


def acked_writes(trace: list[dict]) -> list[tuple[str, bool]]:
    """(service, verified-afterwards) for every success-acked write in an episode."""
    out = []
    for i, s in enumerate(trace):
        service, kind = TOOL_META.get(s["tool"], (None, ""))
        if kind != "write" or service is None or not isinstance(s["obs"], dict) or s["obs"].get("status") != "success":
            continue
        ids = {s["args"].get(k) for k in ID_ARGS} - {None}
        later = trace[i + 1:]
        out.append((service, any(x["tool"] in VERIFICATION_TOOLS and
                                 ids & ({x["args"].get(k) for k in ID_ARGS} - {None}) for x in later)))
    return out


def profile(eps: list[dict]) -> dict[str, tuple[int, int]]:
    c = defaultdict(lambda: [0, 0])
    for e in eps:
        for svc, ver in acked_writes(e["trace"]):
            c[svc][0] += ver
            c[svc][1] += 1
    return {s: (v, n) for s, (v, n) in c.items()}


def spearman(xs, ys) -> float | None:
    from scipy.stats import spearmanr
    if len(set(xs)) < 2 or len(set(ys)) < 2:
        return None
    return float(spearmanr(xs, ys).statistic)


def calibration(eps: list[dict], rho: dict[str, float], boot: int = 2000, seed: int = 0) -> dict:
    import numpy as np
    prof = profile(eps)
    svcs = sorted(s for s in prof if prof[s][1] > 0)
    rate = {s: prof[s][0] / prof[s][1] for s in svcs}
    r = spearman([rate[s] for s in svcs], [rho[s] for s in svcs])
    rng = np.random.default_rng(seed)
    rs = []
    for _ in range(boot):
        b = [eps[i] for i in rng.integers(0, len(eps), len(eps))]
        p = profile(b)
        ss = [s for s in svcs if p.get(s, (0, 0))[1] > 0]
        v = spearman([p[s][0] / p[s][1] for s in ss], [rho[s] for s in ss])
        if v is not None:
            rs.append(v)
    ci = [float(np.percentile(rs, 2.5)), float(np.percentile(rs, 97.5))] if rs else None
    return {"check_rate": {s: {"rate": round(rate[s], 3), "n": prof[s][1], "rho_s": rho[s]} for s in svcs},
            "spearman": r, "spearman_ci95": ci}


def main() -> None:
    for p in sys.argv[1:]:
        eps = [json.loads(l) for l in open(p)]
        rho = SWAPPED_RHO if "calib_swap" in p else TRAIN_RHO
        print(p.rsplit("/", 1)[-1], json.dumps(calibration(eps, rho)))


if __name__ == "__main__":
    main()
