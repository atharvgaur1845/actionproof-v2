"""Derived quantities quoted in the v2 paper, computed from registered result files only.

    python scripts/lab/derive_numbers.py   # -> results/lab/derived_numbers.json
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    s = json.loads((ROOT / "results/lab/lab1_summary.json").read_text())["L1_table"]
    g = json.loads((ROOT / "results/lab/grad_snr.json").read_text())
    l3 = json.loads((ROOT / "results/lab/lab1_summary.json").read_text())["L3G"]
    base = s["nostd/E4"]["n_updates"]
    fork_arms = ["fork_act/E4", "cdvo_fixed@0.25/E4", "cdvo_adapt@0.25/E4", "cdvo_ia@0.25/E4"]
    ratios = {a: base / s[a]["n_updates"] for a in fork_arms}
    crn = [x for v in g["crn_variance_ratio"].values() for x in v[:3]]
    cos_other = [v["cosine_with_exact"] for k, v in g["per"].items() if not k.startswith("grpo")]
    fork_l3 = ["fork_act", "cdvo_adapt@0.25", "cdvo_ia@0.25"]
    gen_ratio = [l3[a][sp] / l3["ppo"][sp] for a in fork_l3 for sp in ("test_shift", "test_long")]
    out = {"episodes_per_update_ratio": ratios, "episodes_per_update_ratio_min": min(ratios.values()),
           "episodes_per_update_ratio_max": max(ratios.values()), "crn_variance_ratio_min": min(crn),
           "crn_variance_ratio_max": max(crn), "min_cosine_non_grpo": min(cos_other),
           "fork_vs_ppo_generalisation_ratio_min": min(gen_ratio), "fork_vs_ppo_generalisation_ratio_max": max(gen_ratio)}
    (ROOT / "results/lab/derived_numbers.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
