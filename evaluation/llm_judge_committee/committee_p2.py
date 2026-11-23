"""P2 secondary (PREREGISTRATION_P2.md §6): the J1 four-family committee on P1/P2 frozen-test episodes. Descriptive;
the deterministic verifier decides every outcome and no judge answer changes a result.

**Reuse.** J1's rubric (variant v1), case renderer (transcript view), prompt builder, router call and parser are
imported unchanged from `committee.py` and `build_cases.py`. The judges are Gemma 4 31B, Qwen3 235B, gpt-oss-120b
and DeepSeek V3.2.

**Sample, fixed before any P2 output.**
- The greedy episode of seed 0 of arms A, C, D (`results/p1/test`) and E, Eg (`results/p2/test`), on all 450 test
  tasks: 2,250 cases.
- Case ids are salted hashes, so judges see neither the arm nor the verdict.
- Truth is kept in `data_p2/truth.jsonl`.

**Budget.** The run stops before the cumulative estimated cost of J1 plus P2 passes the author's $15.

**Reported per arm.** Accuracy of each judge's `operation_occurred` against the verifier's goal_met, with an
abstention counted as not correct. Judge acceptance on verifier failures. The majority of four.

    python evaluation/llm_judge_committee/committee_p2.py build
    python evaluation/llm_judge_committee/committee_p2.py run
    python evaluation/llm_judge_committee/committee_p2.py analyze
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
import build_cases as BC  # noqa: E402
import committee as CM  # noqa: E402

ARMS = {"A": "results/p1/test", "C": "results/p1/test", "D": "results/p1/test", "E": "results/p2/test",
        "Eg": "results/p2/test"}
TEST = ("p1_test_full", "p1_test_pp", "p1_test_rec", "p1_test_wf4", "p1_test_combo")
DATA, OUT = HERE / "data_p2", HERE / "outputs_p2"
SALT = "p2-committee-20261124"
CAP = 15.0


def build() -> None:
    DATA.mkdir(exist_ok=True)
    cases, truth = [], []
    for arm, d in ARMS.items():
        for sp in TEST:
            for r in map(json.loads, (ROOT / d / f"{arm}_s0__{sp}.jsonl").read_text().splitlines()):
                if r["sample"] != -1:
                    continue
                cid = hashlib.sha256(f"{SALT}:{arm}:{r['task_id']}".encode()).hexdigest()[:11]
                r.setdefault("escalated", False)
                cases.append({"case_id": cid, "split": "p2", "views": {"transcript": BC.render(r, with_state=False)}})
                truth.append({"case_id": cid, "arm": arm, "split": sp, "task_id": r["task_id"], **BC.truth(r)})
    cases.sort(key=lambda c: c["case_id"])
    (DATA / "cases.jsonl").write_text("".join(json.dumps(c, ensure_ascii=False) + "\n" for c in cases))
    (DATA / "truth.jsonl").write_text("".join(json.dumps(t) + "\n" for t in truth))
    print(len(cases), "cases")


def spent() -> float:
    return CM.spent() + sum(json.loads(l).get("cost_usd", 0.0) for f in OUT.glob("*.jsonl") for l in f.read_text().splitlines())


def run(workers: int = 8) -> None:
    token = os.environ.get("HF_TOKEN") or Path(os.path.expanduser(os.environ.get(
        "HF_TOKEN_PATH", "/scratch/pabitra/.cache/huggingface/token"))).read_text().strip()
    cases = [json.loads(l) for l in (DATA / "cases.jsonl").read_text().splitlines()]
    OUT.mkdir(exist_ok=True)
    lock, budget = threading.Lock(), {"spent": spent()}
    print(f"spent so far (J1 + P2) ${budget['spent']:.3f}; cap ${CAP}")
    for j in CM.JUDGES:
        model, pin, pout = CM.JUDGES[j][:3]
        out = OUT / f"{j}__transcript__v1__p2.jsonl"
        done = {json.loads(l)["case_id"] for l in out.read_text().splitlines()} if out.exists() else set()

        def work(c: dict) -> None:
            with lock:
                if budget["spent"] >= CAP:
                    return
            resp, dt = CM.call(model, CM.build_prompt(c["views"]["transcript"], c["case_id"], j, "v1"), token)
            u = resp.get("usage") or {}
            cost = u.get("prompt_tokens", 0) * pin / 1e6 + u.get("completion_tokens", 0) * pout / 1e6
            text = ((resp.get("choices") or [{}])[0].get("message", {}) if "choices" in resp else {}).get("content") or ""
            row = {"case_id": c["case_id"], "judge": j, "model": model, "variant": "v1", "parsed": CM.parse(text),
                   "raw": text[:4000], "error": resp.get("error"), "latency_s": round(dt, 2), "cost_usd": cost,
                   "time": time.strftime("%Y-%m-%dT%H:%M:%S")}
            with lock:
                budget["spent"] += cost
                with out.open("a") as fh:
                    fh.write(json.dumps(row, ensure_ascii=False) + "\n")

        with ThreadPoolExecutor(workers) as ex:
            list(ex.map(work, [c for c in cases if c["case_id"] not in done]))
        print(j, "done; total spent", round(budget["spent"], 3))


def analyze() -> None:
    sys.path.insert(0, str(HERE))
    from analyze import q1
    truth = {t["case_id"]: t for t in map(json.loads, (DATA / "truth.jsonl").read_text().splitlines())}
    ans = {j: {r["case_id"]: q1(r["parsed"]) for r in map(json.loads, (OUT / f"{j}__transcript__v1__p2.jsonl").read_text().splitlines())}
           for j in CM.JUDGES if (OUT / f"{j}__transcript__v1__p2.jsonl").exists()}
    res = {}
    for arm in ARMS:
        ids = [i for i, t in truth.items() if t["arm"] == arm]
        fail = [i for i in ids if not truth[i]["goal_met"]]
        row = {"n": len(ids), "verifier_failures": len(fail)}
        for j, a in ans.items():
            row[j] = {"accuracy": sum(a.get(i) == truth[i]["goal_met"] for i in ids) / len(ids),
                      "accept_on_failure": (sum(a.get(i) is True for i in fail) / len(fail)) if fail else None,
                      "answered": sum(a.get(i) is not None for i in ids) / len(ids)}
        maj = {}
        for i in ids:
            v = [a.get(i) for a in ans.values() if a.get(i) is not None]
            maj[i] = (sum(v) > len(v) / 2) if v and sum(v) != len(v) / 2 else None
        row["majority"] = {"accuracy": sum(maj[i] == truth[i]["goal_met"] for i in ids) / len(ids),
                           "accept_on_failure": (sum(maj[i] is True for i in fail) / len(fail)) if fail else None}
        res[arm] = row
    (ROOT / "results/p2").mkdir(parents=True, exist_ok=True)
    (ROOT / "results/p2/committee_p2.json").write_text(json.dumps({"spent_usd_total_j1_p2": spent(), "arms": res}, indent=2))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    {"build": build, "run": run, "analyze": analyze}[sys.argv[1]]()
