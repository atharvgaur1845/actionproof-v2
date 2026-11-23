"""ROI input: the scripted policies on s3_conf_full, scored by the verifier, with tool-call counts.

`oracle` is a hand-written verification layer: read back after every write, then repair with fixed rules.
`naive` trusts every acknowledgement. Both come from actionproof/integrity/sim/agents.py and are never trained.
This reproduces results/s3/scripted_conf_calibration.json (VSC, false success, violation) and adds call counts.

    python scripts/roi_scripted.py   # writes results/roi/scripted_s3_conf_full.json
"""
from __future__ import annotations

import json
from pathlib import Path

from actionproof.bench.generator_proc import load_tasks
from actionproof.integrity.sim import agents
from actionproof.integrity.sim.truth import score
from actionproof.stage3.env import make_env_s3
from actionproof.integrity.sim.runtime import episode_record_proc
from actionproof.stage3 import rewards as R

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    tasks = load_tasks(ROOT / "data/bench/proc/s3_conf_full.jsonl")
    out = {}
    for name in ("oracle", "naive"):
        rows = []
        for i, t in enumerate(tasks):
            env = make_env_s3(t, f"roi-{name}-{i}")
            getattr(agents, name)(env, t)
            rec = episode_record_proc(t, env)
            rec["score"] = s = rec.get("score") or score(rec)
            calls = [c for c in rec["trace"] if c.get("by", "agent") == "agent"]
            rows.append({"vsc": s["vsc"], "false_success": s["false_success"], "violation": bool(R.violation(rec)),  # as in scripts/s3_eval.py
                         "calls": len(calls), "reads": sum(c["tool"].startswith(("get_", "list_", "find_")) for c in calls)})
        n = len(rows)
        out[name] = {k: sum(float(r[k]) for r in rows) / n for k in ("vsc", "false_success", "violation", "calls", "reads")} | {"n": n}
    (ROOT / "results/roi").mkdir(exist_ok=True)
    (ROOT / "results/roi/scripted_s3_conf_full.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
