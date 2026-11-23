"""Data for the website's retention (R1, R1-T), judge (J1) and ROI sections. Built only from result files; a section
whose file does not exist yet gets {"status": "running"} and the page says so. Imported by scripts/build_site.py.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
R1_ARMS = [("a12_rv", "State-reward GRPO", 5), ("a12_ra", "Acknowledgement-reward GRPO", 5),
           ("rsc", "Recovery-start GRPO", 5), ("control", "Class-matched control", 5),
           ("ackonly", "Acknowledgement-only starts", 5), ("sr1", "Tool-name randomisation", 3)]
R1_EPS = [("ifeval_prompt_strict", "IFEval (prompt, strict)"), ("bfcl_nonlive_ast", "BFCL calls (non-live)"),
          ("bfcl_live_ast", "BFCL calls (live)"), ("bfcl_irrelevance", "BFCL: abstains when no tool fits")]


def _j(p: str):
    f = ROOT / p
    return json.loads(f.read_text()) if f.exists() else None


def _r(x, k=4):
    return None if x is None else round(x, k)


def r1() -> dict:
    s = _j("results/r1/r1_summary.json")
    if not s:
        return {"status": "running", "source": "results/r1/r1_summary.json"}
    c = s["contrasts"]

    def row(key):
        x = c.get(key)
        return None if not x else {"d": _r(x["delta"]), "ci": [_r(v) for v in x["ci95"]], "ref": _r(x["ref_value"]),
                                   "val": _r(x["arm_value"]), "seeds": [_r(v) for v in x["per_seed_delta"]]}
    return {"status": "done", "source": "results/r1/r1_summary.json", "margin": s["margin"],
            "eps": R1_EPS, "warm": {e: row(f"warm_vs_base:{e}") for e, _ in R1_EPS},
            "arms": [{"key": a, "label": lab, "seeds": n, "eps": {e: row(f"{a}_vs_warm:{e}") for e, _ in R1_EPS}}
                     for a, lab, n in R1_ARMS],
            "verdicts": {k: s.get(k) for k in ("H1", "H2", "H3")}}


def r1t() -> dict:
    s = _j("results/r1_tau/r1_tau_summary.json")
    pilot = _j("results/r1_tau/pilot/base.summary.json")
    if s:
        return {"status": "done", "source": "results/r1_tau/r1_tau_summary.json", "models": s["models"],
                "contrasts": s["contrasts"]}
    return {"status": "pilot" if pilot else "running", "pilot": pilot, "source": "results/r1_tau/"}


def j1() -> dict:
    s = _j("results/j1/j1_summary.json")
    if not s or "main:transcript" not in s:
        return {"status": "running", "source": "results/j1/j1_summary.json"}
    judges = {"gemma": "Gemma 4 31B", "qwen": "Qwen3 235B", "gptoss": "gpt-oss-120b", "deepseek": "DeepSeek V3.2"}
    out = {"status": "done", "source": "results/j1/j1_summary.json", "rows": []}
    for k, lab in judges.items():
        t = s["main:transcript"]["judges"].get(k, {})
        st = (s.get("main:transcript_state") or {}).get("judges", {}).get(k, {})
        h1 = (s.get("J-H1") or {}).get("per_judge", {}).get(k, {})
        out["rows"].append({"judge": lab, "transcript_acc": _r(t.get("q1_vs_goal_met", {}).get("accuracy_abstain_wrong")),
                            "transcript_cov": _r(t.get("q1_vs_goal_met", {}).get("coverage")),
                            "state_acc": _r(st.get("q1_vs_goal_met", {}).get("accuracy_abstain_wrong")),
                            "fooled": _r(h1.get("share")), "fooled_n": h1.get("n"),
                            "cost_per_1000": _r(t.get("cost_per_1000_usd"), 3), "latency_s": t.get("latency_median_s")})
    out["committee"] = {v: _r(s["main:transcript"].get("committee", {}).get("majority", {}).get(m))
                        for v, m in (("transcript_acc", "accuracy_abstain_wrong"),)}
    out["verdicts"] = {k: (s.get(k) or {}).get("supported") for k in ("J-H1", "J-H2")}
    out["human"] = s.get("human")
    return out


def roi() -> dict:
    s = _j("results/roi/roi_model.json")
    return {"source": "results/roi/roi_model.json", "arms": s["arms"], "assumptions": s["assumptions"],
            "episodes_per_gpu_hour": s["episodes_per_gpu_hour"]}


def build() -> dict:
    return {"r1": r1(), "r1t": r1t(), "j1": j1(), "roi": roi()}


if __name__ == "__main__":
    d = build()
    print({k: v.get("status", "ok") for k, v in d.items()}, len(json.dumps(d)) // 1024, "KB")
