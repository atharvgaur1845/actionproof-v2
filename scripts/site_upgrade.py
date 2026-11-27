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


def p2() -> dict:
    """P2 registered contrasts and references, straight from results/p2/p2_summary.json."""
    src = "results/p2/p2_summary.json"
    s = _j(src)
    if not s or not s.get("primary"):
        return {"status": "running", "source": src}
    rows = [{"id": h, "family": fam, "contrast": r["contrast"], "Y": _r(r.get("Y_mean")), "X": _r(r.get("X_mean")),
             "delta": _r(r["delta"]), "ci95": [_r(c) for c in r["ci95"]], "holm_p": _r(r["holm_p"]),
             "seed_deltas": [_r(x) for x in r.get("seed_deltas", [])], "supported": r["supported"]}
            for fam in ("primary", "secondary") for h, r in s[fam].items()]
    g = _j("results/quality/p2_guard_diagnosis.json") or {}
    pb = g.get("part_b_guarded_oracle", {}).get("p1_test_pp", {})
    return {"status": "done", "source": src, "rows": rows, "references": s.get("references", {}),
            "oracle_pp_blocked": pb.get("tasks_with_a_would_block"), "oracle_pp_tasks": pb.get("tasks")}


def p3() -> dict:
    """P3 registered test (Dn vs D) and the exploratory W/A/Dw/D 2x2, straight from their result files."""
    src, src2 = "results/p3/p3_summary.json", "results/audit/dwarm_2x2.json"
    s, x = _j(src), _j(src2)
    out = {"status": "running", "source": src}
    if s and s.get("H1"):
        h = s["H1"]
        out = {"status": "done", "source": src, "D": _r(h["D_mean"]), "Dn": _r(h["Dn_mean"]), "delta": _r(h["delta"]),
               "ci95": [_r(c) for c in h["ci95"]], "p": _r(h["p_one_sided"]), "seed_deltas": [_r(d) for d in h["seed_deltas"]],
               "supported": h["supported"]}
    if x and x.get("contrasts"):
        out["x2"] = {"source": src2, "arms": {k: _r(v) for k, v in x["arm_means"].items()},
                     "rows": [{"name": k, "delta": _r(v["delta"]), "ci95": [_r(c) for c in v["ci95"]]}
                              for k, v in x["contrasts"].items()]}
    return out


TXN_AGENTS = [("B0", "B0 trust the acknowledgement", "scripted"), ("B1", "B1 final state only", "scripted"),
              ("B3", "B3 bounded retry + idempotency key", "scripted"), ("B4", "B4 reconcile the evidence", "scripted"),
              ("W", "W plain SFT warm start", "trained"), ("Dw", "Dw recovery-SFT warm start", "trained"),
              ("D_s0", "D seed 0 (recovery SFT + GRPO)", "trained"), ("Dn_s0", "Dn seed 0 (no-std GRPO)", "trained"),
              ("aya32b_bf16", "Aya Expanse 32B (prompted)", "prompted")]
QUADS = [("quad:ok_phantom", "ok", "nothing was paid"), ("quad:ok_committed", "ok", "paid"),
         ("quad:timeout_committed", "timeout", "paid"), ("quad:timeout_uncommitted", "timeout", "nothing was paid")]
H3_NAMES = {"dmr": "duplicate mutations", "urr": "unnecessary retries"}


def txn() -> dict:
    """TXN1 (registered): H1 by quadruple reality, per-agent VTC/FCR, the H3 family and H5, from the 1739 output."""
    src = "results/txn/stats__txn_test.json"
    d = _j(src)
    if not d or "tests" not in d.get("H3", {}):
        return {"status": "running", "source": src}
    m = d["models"]
    agents = [{"tag": t, "name": n, "kind": k, "vtc": _r(m[t]["all"]["VTC"]["rate"]), "fcr": _r(m[t]["all"]["FCR"]["rate"]),
               "n": m[t]["all"]["n_episodes"]} for t, n, k in TXN_AGENTS if t in m]
    quads = [{"ack": a, "truth": tr, "fcr": _r(d["H1"]["by_class"][c]["rate"]), "num": d["H1"]["by_class"][c]["num"],
              "den": d["H1"]["by_class"][c]["den"]} for c, a, tr in QUADS if c in d["H1"]["by_class"]]
    tests = [{"metric": H3_NAMES[r["metric"]], "comparator": r["comparator"], "d": _r(r["d_rate"]), "comp": _r(r["comp_rate"]),
              "diff": _r(r["diff"]), "ci95": [_r(c) for c in r["ci95"]], "p_holm": _r(r["p_holm"]),
              "vtc_diff": _r(r["vtc_diff"]["diff"]), "supported": r["supported"]} for r in d["H3"]["tests"]]
    h5 = d["H5"]["per_model"]
    esc = [{"tag": t, "name": n, "cer": _r(h5[t]["CER"]["rate"]), "unsafe": _r(h5[t]["unsafe"]["rate"])}
           for t, n, _ in TXN_AGENTS if t in h5 and t in ("B4", "Dw", "D_s0", "aya32b_bf16")]
    return {"status": "done", "source": src, "n_tasks": 440, "agents": agents, "quads": quads, "tests": tests,
            "n_in": d["H3"]["tests"][0]["n_tasks"], "clusters": d["H3"]["tests"][0]["n_clusters"],
            "supported": d["H3"]["supported"], "h5": esc}


def p4() -> dict:
    """P4 (registered): Dw_aug vs Dw on renamed tools, the unseen four-step workflow and the canonical P1 test."""
    src = "results/p4/p4_summary.json"
    d = _j(src)
    if not d or "H1" not in d:
        return {"status": "running", "source": src}
    names = {"H1": "Renamed tools (150 tasks)", "H2": "Unseen four-step workflow (160 tasks)",
             "H3": "Canonical P1 test, no cost (450 tasks)"}
    return {"status": "done", "source": src, "rows": [
        {"id": h, "name": names[h], "dw": _r(d[h]["Dw"]), "dwaug": _r(d[h]["Dwaug"]), "delta": _r(d[h]["delta"]),
         "ci95": [_r(c) for c in d[h]["ci95"]], "seed_deltas": [_r(x) for x in d[h]["seed_deltas"]],
         "supported": d[h]["supported"]} for h in ("H1", "H2", "H3")]}


def p5() -> dict:
    """P5 (registered): Dw's recovery-SFT recipe on four other Cohere Labs bases."""
    src = "results/p5/p5_summary.json"
    d = _j(src)
    if not d:
        return {"status": "running", "source": src}
    names = {"fire": "Tiny Aya Fire", "water": "Tiny Aya Water", "earth": "Tiny Aya Earth", "expanse8b": "Aya Expanse 8B"}
    rows = [{"name": n, "base": _r(d[k]["base"]), "dw": _r(d[k]["dw"]), "delta": _r(d[k]["delta"]),
             "ci95": [_r(c) for c in d[k]["ci95"]], "supported": d[k]["supported"]} for k, n in names.items() if k in d]
    ref = d.get("reference_tiny_aya_global", {})
    return {"status": "done", "source": src, "rows": rows, "ref": {"base": _r(ref.get("global_base")), "dw": _r(ref.get("global_dw"))}}


def b1api() -> dict:
    """B1 addendum 1 (supplementary, greedy): frontier models on the frozen P1 test splits."""
    rows, splits = [], [("p1_test_full", "full episodes"), ("p1_test_pp", "partial payments"),
                        ("p1_test_rec", "fresh recovery"), ("p1_test_wf4", "four-step workflow"),
                        ("p1_test_combo", "two faults")]
    for tag, name in (("qwen", "Qwen3-235B-A22B"), ("gptoss", "gpt-oss-120b")):
        v = {sp: (_j(f"results/b1/{tag}__{sp}.summary.json") or {}).get("greedy", {}).get("vsc") for sp, _ in splits}
        if any(x is not None for x in v.values()):
            rows.append({"name": name, "vsc": {sp: _r(x) for sp, x in v.items()}})
    return {"status": "done" if rows else "running", "source": "results/b1/", "splits": splits, "rows": rows}


def theory() -> dict:
    """Section 10: exact-theory curves and the registered TH1 estimator comparison (results/theory/site_theory.json)."""
    src = "results/theory/site_theory.json"
    d = _j(src)
    if not d:
        return {"status": "missing", "source": src}
    std = {"grpo", "dapo", "grpo_kl", "gdpo"}
    names = {"grpo": "GRPO", "dapo": "DAPO-style filtering", "grpo_kl": "GRPO + KL", "gdpo": "GDPO-style per-component",
             "nostd": "Dr. GRPO (no std)", "rloo": "RLOO", "reinforce": "REINFORCE + mean baseline", "ppo": "PPO (learned value)",
             "batchstd": "Batch-std scaling", "awrb": "AWRB"}
    return {"status": "done", "source": src, "labels": d["labels"],
            "cost": [{k: r[k] for k in ("p", "c", "grpo", "drgrpo", "bayes")} for r in d["cost"]],
            "th1": [{"key": k, "name": names.get(k, k), "group_std": k in std, "agreement": _r(v["bayes_agreement"]),
                     "regret": _r(v["median_regret"])} for k, v in sorted(d["th1"].items(), key=lambda kv: -kv[1]["bayes_agreement"])]}


def sel1() -> dict:
    """SEL1 (registered): SEL vs CTL on escalation (H1, H2), selectivity J (H3) and resolvable VTC (H4), plus the
    per-group escalation and unsafe rates (descriptive), from the task-1812 output."""
    src = "results/sel1/sel1_summary.json"
    d = _j(src)
    if not d or "SEL1_supported" not in d:
        return {"status": "running", "source": src}
    names = {"H1": "Hands off when no read can settle it: locked TXN test (60 tasks)",
             "H2": "Same, held-out step and workflows (80 tasks; reported on its own)",
             "H3": "Selectivity J: hand-off on unresolvable minus hand-off on degraded-but-resolvable (200 tasks)",
             "H4": "No cost: verified completion on resolvable TXN tasks (380 tasks; non-inferiority)"}
    # 6 decimals: the page shows 3, and rounding twice (4 then 3) can differ from the documents (0.95848 -> 0.959)
    rows = [{"id": h, "name": names[h], "sel": _r(d[h]["J_SEL" if h == "H3" else "SEL"], 6),
             "ctl": _r(d[h]["J_CTL" if h == "H3" else "CTL"], 6), "diff": _r(d[h]["diff"], 6),
             "ci95": [_r(c, 6) for c in d[h]["ci95"]], "p_holm": _r(d[h].get("p_holm")), "bound": d[h].get("bound"),
             "seed_diffs": [_r(x, 6) for x in d[h]["seed_diffs"]], "supported": d[h]["supported"]} for h in ("H1", "H2", "H3", "H4")]
    st = d["descriptive"]["sel1_test"]

    def pooled(arm: str, g: str, m: str) -> dict:
        num = sum(st[f"{arm}_s{k}"][g][m]["num"] for k in range(3))
        den = sum(st[f"{arm}_s{k}"][g][m]["den"] for k in range(3))
        return {"num": num, "den": den, "rate": _r(num / den)}
    groups = [("u_in", "No read can settle it, as in training", True), ("u_receive", "Same, held-out step (receive)", True),
              ("u_ood_wf", "Same, held-out workflows", True), ("t2", "Transient outage, held-out length", False),
              ("audit_down", "Only the audit log is down", False)]
    grp = [{"name": n, "should": should, "sel_esc": pooled("SEL", g, "escalated"), "ctl_esc": pooled("CTL", g, "escalated"),
            "sel_unsafe": pooled("SEL", g, "unsafe"), "ctl_unsafe": pooled("CTL", g, "unsafe")} for g, n, should in groups]
    s2 = st["SEL_s2"]["u_receive"]
    return {"status": "done", "source": src, "rows": rows, "supported": d["SEL1_supported"], "groups": grp,
            "seed2_receive": {"esc": s2["escalated"]["num"], "unsafe": s2["unsafe"]["num"], "den": s2["escalated"]["den"],
                              "ctl_unsafe": st["CTL_s2"]["u_receive"]["unsafe"]["num"]}}


def build() -> dict:
    return {"r1": r1(), "r1t": r1t(), "j1": j1(), "roi": roi(), "p1": p1(), "p2": p2(), "p3": p3(), "theory": theory(),
            "txn": txn(), "sel1": sel1(), "p4": p4(), "p5": p5(), "b1api": b1api()}


if __name__ == "__main__":
    d = build()
    print({k: v.get("status", "ok") for k, v in d.items()}, len(json.dumps(d)) // 1024, "KB")
