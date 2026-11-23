"""ActionProof operational cost model (docs/upgrade/ROI.md). Writes results/roi/roi_model.json.

Two kinds of input, kept apart in the output:

MEASURED (simulator). For every arm and fault class c, this script measures from the recorded `s3_conf_full`
episodes (300 tasks per run, every seed pooled; per-seed ranges kept):
- P(verified completion | c);
- P(false success without an unrefunded duplicate | c);
- P(unrefunded duplicate | c);
- P(honest failure or escalation | c);
- policy calls per episode.
The scripted rows (`results/roi/scripted_s3_conf_full.json`) use the same tasks and the same verifier.

ASSUMED (external). A real deployment has far fewer faults than this fault-enriched pool, so outcome rates are
re-weighted:
    P(outcome) = (1 − f) · P(outcome | clean) + f · Σ_c w_c · P(outcome | c).
- f: the fault incidence per workflow, a scenario assumption;
- w: the pool's own mix over faulty classes;
- costs: the scenario table below, with sources and their quality given in docs/upgrade/ROI.md.

None of this is customer data. Every money figure is a projection from these assumptions.
"""
from __future__ import annotations

import glob
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ARMS = {  # display name -> result-file tag prefix (procurement study, s3_conf_full)
    "SFT agent (warm start)": ["warm_start"],
    "Acknowledgement-reward GRPO (R_A)": ["a12_ra_s"],
    "State-reward GRPO (R_V)": ["a12_rv_s"],
    "Class-matched control GRPO": ["s3_matched_s"],
    "Recovery-start GRPO (RSC)": ["s3_rv_mix_s"],
}
CLASSES = ("clean", "phantom", "partial", "dup_delivery", "lost_ack", "precommit_timeout", "never_commit")

# Scenario assumptions (USD unless noted). Sources and their quality: docs/upgrade/ROI.md §3.
SCENARIOS = {
    "conservative": {"fault_rate": 0.005, "value": 2000, "exception_cost": 25, "late_multiplier": 2,
                     "dup_recovery": 0.70, "gpu_hour": 2.50},
    "moderate": {"fault_rate": 0.02, "value": 5000, "exception_cost": 60, "late_multiplier": 3,
                 "dup_recovery": 0.60, "gpu_hour": 5.00},
    "optimistic": {"fault_rate": 0.05, "value": 10000, "exception_cost": 175, "late_multiplier": 5,
                   "dup_recovery": 0.50, "gpu_hour": 5.00},
}
EPISODES_PER_GPU_HOUR = 9600 * 7.5  # measured Stage 1 throughput (results/roi/throughput.json): 9,600 tickets x 7.5 calls


def outcome(r: dict) -> str:
    if r["vsc"]:
        return "verified"
    if r["violation"]:
        return "unrefunded_duplicate"
    if r["score"]["false_success"]:
        return "false_success"
    return "flagged"  # honest failure, escalation or no report: a human sees it immediately


def measure(prefixes: list[str]) -> dict:
    files = [f for p in prefixes for f in glob.glob(str(ROOT / f"results/s3/conf/{p}*__s3_conf_full.jsonl"))]
    per_seed, pooled = {}, defaultdict(lambda: defaultdict(int))
    calls = defaultdict(list)
    for f in sorted(files):
        seed = Path(f).name.split("__")[0]
        cnt = defaultdict(lambda: defaultdict(int))
        for line in Path(f).read_text().splitlines():
            r = json.loads(line)
            c, o = r["fault_class"], outcome(r)
            cnt[c][o] += 1
            pooled[c][o] += 1
            calls[c].append(r["policy_calls"])
        per_seed[seed] = {c: {o: v / sum(d.values()) for o, v in d.items()} for c, d in cnt.items()}
    rates = {c: {o: v / sum(d.values()) for o, v in d.items()} for c, d in pooled.items()}
    return {"runs": sorted(per_seed), "rates": rates, "per_seed": per_seed,
            "calls": {c: sum(v) / len(v) for c, v in calls.items()}, "n": {c: sum(d.values()) for c, d in pooled.items()}}


def scripted() -> dict:
    """The scripted arms have pool-level rates only; their per-class rates are re-derived below."""
    from actionproof.bench.generator_proc import load_tasks
    from actionproof.integrity.sim import agents
    from actionproof.integrity.sim.runtime import episode_record_proc
    from actionproof.integrity.sim.truth import score
    from actionproof.stage3 import rewards as R
    from actionproof.stage3.env import make_env_s3
    tasks = load_tasks(ROOT / "data/bench/proc/s3_conf_full.jsonl")
    out = {}
    for name, label in (("naive", "No verification (scripted, trusts every acknowledgement)"),
                        ("oracle", "Hand-written database checks + fixed repair rules (scripted)")):
        pooled, calls = defaultdict(lambda: defaultdict(int)), defaultdict(list)
        for i, t in enumerate(tasks):
            env = make_env_s3(t, f"roi-{name}-{i}")
            getattr(agents, name)(env, t)
            rec = episode_record_proc(t, env)
            rec["score"] = score(rec)
            rec["vsc"], rec["violation"] = rec["score"]["vsc"], bool(R.violation(rec))
            c = t.meta["fault_class"] if "fault_class" in t.meta else rec.get("fault_class")
            pooled[c][outcome(rec)] += 1
            calls[c].append(sum(1 for x in rec["trace"] if x.get("by", "agent") == "agent"))
        out[label] = {"runs": [name], "rates": {c: {o: v / sum(d.values()) for o, v in d.items()} for c, d in pooled.items()},
                      "calls": {c: sum(v) / len(v) for c, v in calls.items()},
                      "n": {c: sum(d.values()) for c, d in pooled.items()}, "llm": False}
    return out


def project(m: dict, s: dict, llm: bool = True) -> dict:
    faulty = [c for c in CLASSES if c != "clean" and c in m["rates"]]
    w = {c: m["n"][c] / sum(m["n"][x] for x in faulty) for c in faulty}

    def p(o: str) -> float:
        clean = m["rates"].get("clean", {}).get(o, 0.0)
        return (1 - s["fault_rate"]) * clean + s["fault_rate"] * sum(w[c] * m["rates"][c].get(o, 0.0) for c in faulty)
    calls = (1 - s["fault_rate"]) * m["calls"].get("clean", 0) + s["fault_rate"] * sum(w[c] * m["calls"][c] for c in faulty)
    per1000 = {o: 1000 * p(o) for o in ("verified", "false_success", "unrefunded_duplicate", "flagged")}
    cost = {
        "false_success_usd": per1000["false_success"] * s["exception_cost"] * s["late_multiplier"],
        "duplicate_usd": per1000["unrefunded_duplicate"] * (s["value"] * (1 - s["dup_recovery"])
                                                            + s["exception_cost"] * s["late_multiplier"]),
        "flagged_usd": per1000["flagged"] * s["exception_cost"],
        "inference_usd": (1000 * calls / EPISODES_PER_GPU_HOUR * s["gpu_hour"]) if llm else 0.0,
    }
    cost["total_usd"] = sum(cost.values())
    return {"per_1000_ops": per1000, "calls_per_op": calls, "cost_per_1000_ops": cost,
            "cost_per_1000_completed_usd": cost["total_usd"] / per1000["verified"] * 1000 if per1000["verified"] else None}


def main() -> None:
    arms = {name: measure(pref) | {"llm": True} for name, pref in ARMS.items()}
    arms |= scripted()
    out = {"note": "Simulator frequencies are measured; every money figure is a projection from the assumptions.",
           "assumptions": SCENARIOS, "episodes_per_gpu_hour": EPISODES_PER_GPU_HOUR, "arms": {}, "projections": {}}
    for name, m in arms.items():
        out["arms"][name] = {k: m[k] for k in ("runs", "rates", "calls", "n", "llm")}
        out["projections"][name] = {sc: project(m, s, m["llm"]) for sc, s in SCENARIOS.items()}
    base = "SFT agent (warm start)"
    for sc in SCENARIOS:
        b = out["projections"][base][sc]["cost_per_1000_ops"]["total_usd"]
        for name in arms:
            pr = out["projections"][name][sc]
            pr["saving_vs_sft_agent_per_1000_usd"] = b - pr["cost_per_1000_ops"]["total_usd"]
    # sensitivity: moderate scenario, one assumption at a time from the conservative to the optimistic value
    sens = {}
    for k in ("fault_rate", "value", "exception_cost", "late_multiplier", "dup_recovery", "gpu_hour"):
        sens[k] = {}
        for end in ("conservative", "optimistic"):
            s = dict(SCENARIOS["moderate"], **{k: SCENARIOS[end][k]})
            sens[k][end] = {name: project(m, s, m["llm"])["cost_per_1000_ops"]["total_usd"] for name, m in arms.items()}
    out["sensitivity_moderate"] = sens
    (ROOT / "results/roi").mkdir(exist_ok=True)
    (ROOT / "results/roi/roi_model.json").write_text(json.dumps(out, indent=2))
    for sc in SCENARIOS:
        print(f"== {sc}")
        for name in arms:
            pr = out["projections"][name][sc]
            print(f"  {name[:58]:58s} FS/1k {pr['per_1000_ops']['false_success']:6.2f} dup/1k "
                  f"{pr['per_1000_ops']['unrefunded_duplicate']:5.2f} cost/1k ${pr['cost_per_1000_ops']['total_usd']:9.2f}")


if __name__ == "__main__":
    main()
