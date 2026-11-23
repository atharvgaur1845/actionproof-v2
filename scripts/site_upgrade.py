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
    st = s.get("main:transcript_state", {})
    out["state_committee"] = _r(st.get("committee", {}).get("majority", {}).get("accuracy_abstain_wrong"))
    out["all_four_wrong_state"] = len(st.get("all_four_contradict_verifier", []))
    out["max_fooled"] = max((v["fooled"] for v in (s.get("J-H1") or {}).get("per_judge", {}).values()), default=None)
    out["n_unverifiable"] = next(iter((s.get("J-H1") or {}).get("per_judge", {}).values()), {}).get("n")
    a1 = _j("results/j1/j1_addendum1.json")
    out["addendum1"] = a1
    out["human"] = s.get("human")
    return out


def roi() -> dict:
    s = _j("results/roi/roi_model.json")
    return {"source": "results/roi/roi_model.json", "arms": s["arms"], "assumptions": s["assumptions"],
            "episodes_per_gpu_hour": s["episodes_per_gpu_hour"]}


P1_ARMS = {"A": "Recovery-start GRPO (reference)", "B": "A + reconciliation line at inference",
           "C": "A + recovery SFT", "D": "Recovery SFT warm start + the same GRPO"}
P1_TEST = ("p1_test_full", "p1_test_pp", "p1_test_rec", "p1_test_wf4", "p1_test_combo")
P1_SHOW = ("p1_test_pp", "p1_test_wf4")  # browsable same-task episodes: unseen pay ratios and the unseen workflow
P1_WRITES = ("place_supplier_order", "pay_supplier", "record_goods_receipt", "post_journal_entry", "refund_payment")


def _usd(c):
    return None if c is None else f"${c / 100:,.2f}"


def _args(tool: str, a: dict) -> str:
    bits = []
    if "amount_cents" in a:
        bits.append(_usd(a["amount_cents"]))
    if "qty" in a:
        bits.append(f"qty {a['qty']}")
    if "payment_id" in a:
        bits.append(a["payment_id"])
    if "idempotency_key" in a:
        bits.append(f"key {a['idempotency_key']}")
    if tool == "submit_report":
        bits.append(str(a.get("status")))
    return ", ".join(bits)


def _ack(tool: str, o: dict) -> str:
    if o.get("status") != "ok":
        return f"{o.get('status')}: {o.get('error', '')}".strip(": ")
    if tool == "list_payments":
        ps = o.get("payments", [])
        refunded = {p["refund_of"] for p in ps if p.get("kind") == "refund"}
        live = [p for p in ps if p.get("kind") == "payment" and p["payment_id"] not in refunded]
        return f"{len(live)} live payment(s), {_usd(sum(p['amount_cents'] for p in live))}" + \
            (f"; {len(refunded)} refunded" if refunded else "")
    if tool == "list_goods_receipts":
        return f"received qty {sum(r.get('qty', 0) for r in o.get('receipts', []))}"
    for k in ("payment_id", "refund_id", "receipt_id", "entry_id", "order_ref"):
        if k in o:
            return f"ok, {o[k]}"
    return "ok"


def _episode(r: dict, n_prefix: int) -> dict:
    att = {x["idx"]: x for x in r.get("attempts", [])}
    steps = []
    for s in r["trace"]:
        a = att.get(s["idx"])
        db = None
        if s["tool"] in P1_WRITES and a:
            db = a["committed"] + (" (deduplicated: same key)" if a.get("from_idempotency_store") else "")
        steps.append([s["tool"], _args(s["tool"], s["args"]), _ack(s["tool"], s["obs"]), db, s["idx"] < n_prefix])
    sc = r["score"]
    return {"steps": steps, "vsc": r["vsc"], "false_success": sc["false_success"], "verdicts": sc["steps"],
            "violation": bool(r["violation"]), "report": (r.get("report") or {}).get("status"), "ended_by": r["ended_by"]}


def p1() -> dict:
    s = _j("results/p1/p1_summary.json")
    src = "results/p1/p1_summary.json"
    if not s or not s.get("test"):
        return {"status": "running", "source": src, "dev": (s or {}).get("dev") or None}
    import importlib.util
    spec = importlib.util.spec_from_file_location("p1_eval", Path(__file__).with_name("p1_eval.py"))
    pe = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pe)
    union, splits = {}, {}
    for arm in P1_ARMS:
        rows = []
        for sp in P1_TEST:
            for k in (0, 1, 2):
                rows += [json.loads(l) for l in (ROOT / f"results/p1/test/{arm}_s{k}__{sp}.jsonl").read_text().splitlines()]
        keep = ("vsc", "false_success", "duplicate_unrefunded", "partial_pay_repaired", "new_key_share_on_repay",
                "redundant_reads_per_episode", "reads_per_episode", "calls_per_success", "context_ended")
        sm = pe.summarise(rows)
        union[arm] = {m: {mode: _r(sm[mode][m]) for mode in ("greedy", "sampled")} for m in keep}
        union[arm]["by_workflow"] = {w: _r(v["vsc"]) for w, v in sm["by_workflow_sampled"].items()}
        splits[arm] = {sp: _r(s["descriptive"][arm]["test"].get(sp, {}).get("sampled", {}).get("vsc")) for sp in P1_TEST}
    tasks = {}
    for sp in P1_SHOW:
        meta = {t["task_id"]: t for t in map(json.loads, (ROOT / f"data/bench/proc/{sp}.jsonl").read_text().splitlines())}
        for arm in P1_ARMS:
            for r in map(json.loads, (ROOT / f"results/p1/test/{arm}_s0__{sp}.jsonl").read_text().splitlines()):
                if r["sample"] != -1:
                    continue
                t = meta[r["task_id"]]
                d = tasks.setdefault(r["task_id"], {"split": sp, "workflow": t["workflow"], "prompt": t["prompt"],
                                                    "faults": [f"{f['kind']} on {f['tool']}" + (f" (ratio {f['ratio']})" if "ratio" in f else "")
                                                               for f in t["meta"]["faults"]], "arms": {}})
                d["arms"][arm] = _episode(r, len(t["meta"].get("start_prefix") or []))
    return {"status": "done", "source": src, "arms": P1_ARMS, "test": s["test"], "dev": s["dev"], "union": union,
            "splits": splits, "tasks": dict(sorted(tasks.items())), "show": P1_SHOW}


def build() -> dict:
    return {"r1": r1(), "r1t": r1t(), "j1": j1(), "roi": roi(), "p1": p1()}


if __name__ == "__main__":
    d = build()
    print({k: v.get("status", "ok") for k, v in d.items()}, len(json.dumps(d)) // 1024, "KB")
