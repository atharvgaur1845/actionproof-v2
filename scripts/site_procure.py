"""Data for the website's procurement chapter (Study 2: ProcureSim, finalist round), built from the real simulator and
the registered result files only. Imported by scripts/build_site.py.

  sim      every simulator scenario (workflow x fault x faulted step x partial ratio) run by two scripted agents
           (trusting: never reads; verifying: the reference oracle) through the real ProcureSim environment and
           verifier, precomputed here. No model runs in the browser.
  replays  recorded greedy evaluation episodes of five trained models on the same fresh confirmatory tasks
           (product/data/replays.json, built by product/build_data.py from results/s3/conf)
  results  registered and descriptive results, each with its source file
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def j(p: str):
    return json.loads((ROOT / p).read_text())


def _short(v):
    if isinstance(v, dict):
        return {k: _short(x) for k, x in v.items() if k not in ("summary",)}
    if isinstance(v, list):
        return [_short(x) for x in v[:4]]
    return v


def sim() -> dict:
    from actionproof.bench import generator_proc as G
    from product.engine import FAULTS, WORKFLOWS, Scenario, run_policy
    out = {"workflows": WORKFLOWS, "faults": FAULTS, "runs": {}}
    for wf, steps in G.WORKFLOWS.items():
        for fault in FAULTS:
            for step in [s for s in steps if s != "post"]:
                if fault == "partial" and step not in ("pay", "receive"):
                    continue
                if fault == "dup_delivery" and step != "pay":
                    continue
                if fault == "none" and step != [s for s in steps if s != "post"][0]:
                    continue
                for ratio in ((0.25, 0.5, 0.75) if fault == "partial" else (0.5,)):
                    sc = Scenario(workflow=wf, fault=fault, fault_step=step, ratio=ratio)
                    key = f"{wf}|{fault}|{step}|{ratio}"
                    runs = {}
                    for pol in ("trusting", "verifying"):
                        r = run_policy(sc, pol)
                        rows = [{"tool": x["visible"]["call"]["tool"], "args": _short(x["visible"]["call"]["args"]),
                                 "resp": _short(x["visible"]["response"]),
                                 "db": {t: [_short(rr) for rr in rs] for t, rs in x["state"]["new_rows"].items()},
                                 "commit": x["state"]["commit"], "div": x["truth"]["diverged"]} for x in r["timeline"]]
                        rw = r["rewards"]
                        runs[pol] = {"rows": rows, "verdicts": rw["evidence"]["step_verdicts"],
                                     "report": (rw["evidence"]["report"] or {}).get("status"),
                                     "rv": rw["R_V (state reward)"], "ra": rw["R_A (acknowledgement reward)"],
                                     "vsc": rw["VSC (evaluation metric)"], "false": rw["evidence"]["false_completion"],
                                     "dup": rw["evidence"]["duplicate"]}
                    t = r["task"]
                    out["runs"][key] = {"prompt": t["prompt"], "po": t["meta"]["po_id"], **runs}
    return out


def replays() -> dict:
    d = j("product/data/replays.json")
    keep = []
    for t in d["tasks"]:
        runs = {}
        for m, r in t["runs"].items():
            runs[m] = {"calls": [{"tool": x["tool"], "args": _short({k: v for k, v in (x.get("args") or {}).items()
                                                                     if k != "summary"}),
                                  "resp": _short(x.get("obs") or {})} for x in r["trace"] if x.get("by") == "agent"][:12],
                       "report": (r.get("report") or {}).get("status"), "vsc": r["score"]["vsc"],
                       "false": r["score"]["false_success"], "violation": bool(r.get("violation"))}
        keep.append({"id": t["task_id"], "fault": t["fault_class"], "workflow": t["workflow"], "prompt": t["prompt"],
                     "runs": runs})
    return {"models": d["models"], "tasks": keep}


def results() -> dict:
    a12 = j("results/s2/a12_summary.json")
    s3, s3a = j("results/s3/s3_summary.json"), j("results/s3/s3_addendum.json")
    rv = j("results/s3/s3_review_numbers.json")
    sl = j("results/s3/s3_seedlevel.json")
    s4, d4 = j("results/s4/s4_summary.json"), j("results/s4/s4_diagnostics.json")
    fca = j("results/s4/fca_dev_summary.json")
    wf = j("results/s2/partial_by_workflow.json")["a15"]

    def across(d, sp):
        a = d["splits"][sp]["across"]["vsc"]
        return {"est": a["seed_avg"]["est"], "ci": a["seed_avg"]["ci95"], "per_seed": a["per_seed"]}
    shifts = []
    for sp, lab in (("s4_lang_en", "English (reference)"), ("s4_lang_hi", "Hindi request"),
                    ("s4_lang_hinglish", "Hinglish request"), ("s4_lang_para", "Reworded English"),
                    ("s4_combo", "Two faults in one episode"), ("s4_api", "Renamed tools and replies"),
                    ("s4_wf4", "New four-step workflow")):
        c = s4["splits"][sp]["rsc_minus_rv"]["vsc"]
        shifts.append({"split": sp, "label": lab, "est": c["seed_avg"]["est"], "ci": c["seed_avg"]["ci95"],
                       "npos": c["n_positive"], "levels": d4["vsc_per_arm"][sp]})
    fam = {g: {"supported": v["supported"], "p_holm": v["p_holm"], "splits": v["splits"]} for g, v in s4["family"].items()}
    return {
        "a12": {"readback_rv": [j(f"results/eval/a12/conf/a12_proc_rv_s{n}__neutral__proc_conf.summary.json")["phantom_readback"]
                                for n in range(5)],
                "readback_ra": [j(f"results/eval/a12/conf/a12_proc_ra_s{n}__neutral__proc_conf.summary.json")["phantom_readback"]
                                for n in range(5)],
                "vsc_diff": {k: v["diff"] for k, v in a12["vsc"]["per_seed"].items()},
                "src": "results/s2/a12_summary.json; results/eval/a12/conf/"},
        "use": {m: {w: wf[m][w]["rate"] for w in ("pay_post", "order_prepay", "receive_pay")}
                for m in ("warm_start", "rv_s0", "rv_s1", "rv_s2", "rv_s3", "rv_s4")},
        "s3": {"recipe_rec": across(s3, "s3_conf_rec"), "recipe_full": across(s3, "s3_conf_full"),
               "start_rec": across(s3a, "s3_conf_rec"),
               "start_pd": across(j("results/s3/s3_addendum_full_pd.json"), "s3_conf_full"),
               "viol_pd": j("results/s3/s3_addendum_full_pd.json")["splits"]["s3_conf_full"]["across"]["violation"]["seed_avg"],
               "stress_vs_ctl": s3a["splits"]["s3_conf_stress"]["across"]["vsc"]["seed_avg"],
               "seed_p": sl["splits"]["s3_conf_rec"]["rsc_minus_rv"]["vsc"]["unpaired_perm_p_one_sided"],
               "by_class": {k: v["contribution"] for k, v in sl["splits"]["s3_conf_full"]["rsc_minus_rv_by_class"].items()},
               "learned": {k: [v["runs_ge_040"], v["runs"]] for k, v in rv["learned"].items()},
               "rp_cell": {k: [v["repaired"], v["n"]] for k, v in rv["rp_cell"]["s3_conf_full"].items()},
               "src": "results/s3/s3_summary.json; s3_addendum.json; s3_seedlevel.json; s3_review_numbers.json"},
        "shifts": shifts, "family": fam, "diag": d4["per_arm"],
        "transfer": {a: {"vsc": v["vsc_mean"], "col": v["collateral_mean"]}
                     for a, v in rv["transfer"]["care_admin"].items() if isinstance(v, dict)},
        "fca": {"diff": fca["criteria"]["s3_rec_dev_diff"], "per_seed": fca["per_seed"]},
    }


MODELS = {"warm_start": "Before training (warm start)", "a12_ra_s0": "Trained on the acknowledgement reward",
          "a12_rv_s0": "Trained on the state reward", "s3_matched_s0": "State reward, full-episode control",
          "s3_rv_mix_s0": "State reward + recovery starts"}
STORIES = {"s3_conf_full-0001": "phantom", "s3_conf_full-0002": "partial_pay_first",
           "s3_conf_full-0014": "partial_after_receipt"}


def story(tid: str) -> dict:
    """One recorded confirmatory task under five models (seed 0): every call, reply, report, score and the payments
    stored for the task's purchase order at the end (results/s3/conf/<model>__s3_conf_full.jsonl)."""
    out = {"id": tid, "src": "results/s3/conf/<model>__s3_conf_full.jsonl", "models": {}}
    for m, label in MODELS.items():
        r = next(json.loads(line) for line in (ROOT / f"results/s3/conf/{m}__s3_conf_full.jsonl").open()
                 if f'"{tid}"' in line and json.loads(line)["task_id"] == tid)
        t = r["task"]
        po = next(p for p in t["initial_state"]["pos"] if p["po_id"] == t["meta"]["po_id"])
        out.update(prompt=t["prompt"], workflow=t["workflow"], fault=t["meta"]["faults"], po=po["po_id"],
                   amount=po["amount_cents"])
        out["models"][m] = {"label": label, "vsc": r["score"]["vsc"], "false": r["score"]["false_success"],
                            "report": r["report"], "steps": r["score"]["steps"],
                            "calls": [{"tool": x["tool"], "args": {k: v for k, v in x["args"].items() if k != "summary"},
                                       "resp": _short(x["obs"])} for x in r["trace"] if x.get("by") == "agent"],
                            "paid": [p["amount_cents"] if p["kind"] == "payment" else -p["amount_cents"]
                                     for p in r["final_state"]["payments"] if p["po_id"] == po["po_id"]]}
    return out


def build() -> dict:
    gate = j("results/eval/a12/init__neutral__proc_train_T1x4.summary.json")["phantom_readback"]
    res = results()
    res["a12"]["warm_sampled_readback"] = gate
    return {"sim": sim(), "stories": {k: story(t) for t, k in STORIES.items()}, "results": res}


if __name__ == "__main__":
    d = build()
    s = json.dumps(d, separators=(",", ":"))
    print(len(d["sim"]["runs"]), "scenarios;", len(s) // 1024, "KB")
