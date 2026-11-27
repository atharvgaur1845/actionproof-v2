"""RG2 registered analysis (PREREGISTRATION_RG2.md). Written and hashed before any rg2_test output exists.

Arms on rg2_test (240 tasks; groups u_in, u_receive, u_ood_wf = U, unresolvable, 120 tasks; t2 and audit_down = R,
resolvable outages, 80 tasks; std, 40 tasks): V2_D_s{k} = D seed k behind gate v2; V1_D_s{k} = behind gate v1 (the
frozen RG1 gate); k = 0, 1, 2; 1 greedy + 2 sampled episodes per task. Unit, statistic and bootstrap as RG1
(scripts/rg1_stats.py: Boot, compare, holm; B = 10,000), rng seed 20261129.

- H1 selectivity: J = esc(U) - esc(R); J(V2) - J(V1) > 0. U and R are resampled independently.
- H2 transient outages: STR on t2 (40 tasks), V2 - V1 > 0.
- H1, H2: supported iff Holm-adjusted one-sided p < 0.05 (Holm over the two) AND all three seed differences > 0.
- H3 (non-inferiority, outside the family): esc(U), V2 - V1; supported iff the lower 95% bound > -0.03.
RG2 is supported overall iff H1, H2 and H3 all hold.
Descriptive: per-group counts (escalated, STR, VTC, unsafe, FCR) for every tag in results/rg2 on rg2_test.

    python scripts/rg2_stats.py      # writes results/rg2/rg2_summary.json and .md
"""
from __future__ import annotations

import importlib.util
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
_spec = importlib.util.spec_from_file_location("rg1_stats", ROOT / "scripts/rg1_stats.py")
S = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(S)
R = ROOT / "results/rg2"
SEED, SEEDS = 20261129, (0, 1, 2)
U, RG, GROUPS = ("u_in", "u_receive", "u_ood_wf"), ("t2", "audit_down"), ("u_in", "u_receive", "u_ood_wf", "t2",
                                                                           "audit_down", "std")


def main() -> None:
    split = "rg2_test"
    meta = S.task_meta(split)
    v2 = [S.load(f"V2_D_s{k}", split, R) for k in SEEDS]
    v1 = [S.load(f"V1_D_s{k}", split, R) for k in SEEDS]
    miss = [f"{p}_D_s{k}" for p, arm in (("V2", v2), ("V1", v1)) for k, d in zip(SEEDS, arm)
            if d is None or len(d) != len(meta)]
    if miss:
        sys.exit(f"missing or incomplete confirmatory files: {miss}")
    rng = np.random.default_rng(SEED)
    tu = sorted(t for t in meta if meta[t]["group"] in U)
    tr = sorted(t for t in meta if meta[t]["group"] in RG)
    t2 = sorted(t for t in meta if meta[t]["group"] == "t2")
    bu, br, b2 = S.Boot(tu, meta, rng), S.Boot(tr, meta, rng), S.Boot(t2, meta, rng)
    res: dict = {"B": S.B, "seed": SEED, "n_tasks": {"U": len(tu), "R": len(tr), "t2": len(t2)}}
    su2 = [S.scores(x, tu, "escalated") for x in v2]
    su1 = [S.scores(x, tu, "escalated") for x in v1]
    sr2 = [S.scores(x, tr, "escalated") for x in v2]
    sr1 = [S.scores(x, tr, "escalated") for x in v1]
    j2 = [a.mean() - b.mean() for a, b in zip(su2, sr2)]
    j1 = [a.mean() - b.mean() for a, b in zip(su1, sr1)]
    bs = bu.means(np.mean(su2, 0) - np.mean(su1, 0)) - br.means(np.mean(sr2, 0) - np.mean(sr1, 0))
    res["H1"] = {"statistic": "J = esc(U) - esc(t2, audit_down)", "J_V2": float(np.mean(j2)), "J_V1": float(np.mean(j1)),
                 "esc_U": {"V2": float(np.mean(su2)), "V1": float(np.mean(su1))},
                 "esc_R": {"V2": float(np.mean(sr2)), "V1": float(np.mean(sr1))},
                 "diff": float(np.mean(j2) - np.mean(j1)), "ci95": [float(np.quantile(bs, .025)), float(np.quantile(bs, .975))],
                 "p_one_sided": S.p_one_sided(bs), "seed_diffs": [float(a - b) for a, b in zip(j2, j1)]}
    res["H2"] = S.compare(v2, v1, t2, "str", b2)
    ph = S.holm({h: res[h]["p_one_sided"] for h in ("H1", "H2")})
    for h in ("H1", "H2"):
        res[h]["p_holm"] = ph[h]
        res[h]["all_seed_diffs_positive"] = all(x > 0 for x in res[h]["seed_diffs"])
        res[h]["supported"] = bool(ph[h] < 0.05 and res[h]["all_seed_diffs_positive"])
    res["H3"] = S.compare(v2, v1, tu, "escalated", bu)
    res["H3"].update(bound=-0.03, supported=bool(res["H3"]["ci95"][0] > -0.03))
    res["RG2_supported"] = bool(res["H1"]["supported"] and res["H2"]["supported"] and res["H3"]["supported"])
    desc = {}
    for f in sorted(R.glob(f"*__{split}.jsonl")):
        by: dict[str, list] = defaultdict(list)
        for line in f.read_text().splitlines():
            if line.strip():
                t = json.loads(line)["txn"]
                by[t["group"]].append(t)
        desc[f.name.split("__")[0]] = {g: {fl: [sum(bool(e[fl]) for e in by[g]), len(by[g])]
                                           for fl in ("escalated", "str", "vtc", "unsafe", "fcr")} for g in GROUPS}
    res["descriptive"] = desc
    (R / "rg2_summary.json").write_text(json.dumps(res, indent=1, sort_keys=True) + "\n")
    L = ["# RG2 registered analysis (scripts/rg2_stats.py)", "", f"rg2_test: {len(meta)} tasks. V2 = D behind gate v2 "
         "(bounded re-read), V1 = behind the RG1 gate; seeds 0-2; 3 episodes per task.", "",
         "| Hypothesis | V2 | V1 | Difference [95% CI] | Holm p | Seed differences | Supported |", "|---|---|---|---|---|---|---|"]
    h = res["H1"]
    L.append(f"| H1 selectivity J | {h['J_V2']:.3f} | {h['J_V1']:.3f} | {h['diff']:.3f} [{h['ci95'][0]:.3f}, {h['ci95'][1]:.3f}] | "
             f"{h['p_holm']:.4f} | {', '.join(f'{x:.3f}' for x in h['seed_diffs'])} | {'yes' if h['supported'] else 'no'} |")
    for k, name in (("H2", "H2 STR on t2"), ("H3", "H3 escalation on U (bound -0.03)")):
        h = res[k]
        L.append(f"| {name} | {h['a']:.3f} | {h['b']:.3f} | {h['diff']:.3f} [{h['ci95'][0]:.3f}, {h['ci95'][1]:.3f}] | "
                 f"{h.get('p_holm', float('nan')):.4f} | {', '.join(f'{x:.3f}' for x in h.get('seed_diffs', []))} | "
                 f"{'yes' if h['supported'] else 'no'} |")
    L += ["", f"**RG2 supported overall: {'yes' if res['RG2_supported'] else 'no'}.**", "",
          "## Descriptive (escalated / STR / unsafe counts over episodes)", "", "| Tag | " + " | ".join(GROUPS) + " |",
          "|---|" + "---|" * len(GROUPS)]
    for tag, d in desc.items():
        L.append(f"| {tag} | " + " | ".join(f"esc {d[g]['escalated'][0]}/{d[g]['escalated'][1]}, STR {d[g]['str'][0]}, "
                                            f"unsafe {d[g]['unsafe'][0]}" for g in GROUPS) + " |")
    (R / "rg2_summary.md").write_text("\n".join(L) + "\n")
    print("\n".join(L[:9]))


if __name__ == "__main__":
    main()
