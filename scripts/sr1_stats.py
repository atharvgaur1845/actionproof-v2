"""SR1 statistics (PREREGISTRATION_SR1): SR (`s3_sr_mix`) minus RSC (`s3_rv_mix`), seeds 0-2, paired by seed and task.
SR-H1: `s4_api` VSC (3/3 positive and seed-averaged 95% interval > 0). SR-H2: `s3_conf_full` lower bound > -0.05.
    python scripts/sr1_stats.py
"""
from __future__ import annotations

import importlib.util
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("s4_stats", ROOT / "scripts/s4_stats.py")
S4 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(S4)
SPLITS = {"s4_api": "results/s4/eval", "s4_lang_en": "results/s4/eval", "s4_lang_hi": "results/s4/eval",
          "s4_lang_hinglish": "results/s4/eval", "s4_wf4": "results/s4/eval", "s4_combo": "results/s4/eval",
          "s3_conf_full": "results/s3/conf", "s3_conf_rec": "results/s3/conf"}


def main() -> None:
    rng = random.Random(0)
    res: dict = {"splits": {}, "missing": []}
    for sp, d in SPLITS.items():
        sr = [S4.load(f"s3_sr_mix_s{s}", sp, d) for s in range(3)]
        rsc = [S4.load(f"s3_rv_mix_s{s}", sp, d) for s in range(3)]
        res["missing"] += [f"{p}_s{s}__{sp}" for p, ms in (("s3_sr_mix", sr), ("s3_rv_mix", rsc))
                           for s, m in enumerate(ms) if m is None]
        if all(sr) and all(rsc):
            res["splits"][sp] = {k: S4.contrast(sr, rsc, k, rng) for k in ("vsc", "violation")}
    a, f = res["splits"].get("s4_api", {}).get("vsc"), res["splits"].get("s3_conf_full", {}).get("vsc")
    if a:
        res["SR-H1"] = {"supported": a["n_positive"] == 3 and a["seed_avg"]["ci95"][0] > 0, **{k: a[k] for k in (
            "per_seed", "seed_avg", "unpaired_perm_p_one_sided")}}
    if f:
        res["SR-H2"] = {"supported": f["seed_avg"]["ci95"][0] > -0.05, "seed_avg": f["seed_avg"]}
    (ROOT / "results/s4/sr1_summary.json").write_text(json.dumps(res, indent=2))
    print({k: res.get(k) for k in ("SR-H1", "SR-H2")}, "missing", len(res["missing"]))


if __name__ == "__main__":
    main()
