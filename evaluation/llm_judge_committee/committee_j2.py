"""J2 judge runs (PREREGISTRATION_J2.md): Aya Expanse 32B as the historical baseline judge, answer-order reversal,
repeated runs and a paraphrase run on a stratified 200-case subset of the J1 main set.

Reuses J1's prompt, parser and HTTP client unchanged (committee.py). Outputs go to `outputs_j2/` so that J1's files
and P2's committee files are never touched.

**Budget (hard stop).** Before every call the cumulative token-priced estimate over `outputs_j2/*.jsonl` is compared
with `--cap` (default $4.50, inside the $5 the lead engineer allotted to this study). Aya Expanse is priced
conservatively at $0.60 / 1M input and $2.00 / 1M output (no router price is listed for the cohere provider).

    python evaluation/llm_judge_committee/committee_j2.py --run aya
    python evaluation/llm_judge_committee/committee_j2.py --run subset
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from committee import JUDGES, PREAMBLE, QUESTIONS, build_prompt, call, parse  # noqa: E402

J2_JUDGES = {**JUDGES, "aya": ("CohereLabs/aya-expanse-32b:cohere", 0.60, 2.00, "CC-BY-NC-4.0 (model card)",
                               "Cohere Aya (same family as the policy)")}
OUT = HERE / "outputs_j2"
SUBSET_SEED = 20261124
SUBSET_N = 200


def build_prompt_reversed(case_text: str, case_id: str, judge: str) -> str:
    """v1 with every question's answer options in the reverse of the order v1 showed this (case, judge)."""
    rng = random.Random(f"{case_id}:{judge}:v1")
    qs = []
    for key, q, opts in QUESTIONS:
        o = opts[:]
        rng.shuffle(o)
        o = o[::-1]
        qs.append(f'- "{key}": {q} One of: {", ".join(o)}.')
    return (f"{PREAMBLE['v1']}\n\n=== EPISODE ===\n{case_text}\n=== END OF EPISODE ===\n\n"
            "Answer with one JSON object and nothing else, with these keys:\n" + "\n".join(qs) +
            '\n- "confidence": a number from 0 to 1, your probability that your operation_occurred answer is correct.'
            '\n- "rationale": at most 60 words.')


def subset_ids() -> list[str]:
    """Stratified subset of the J1 main set: proportional to stratum size, at least 4 per stratum, seed fixed."""
    truth = [json.loads(l) for l in (HERE / "data/truth.jsonl").read_text().splitlines()]
    main = [t for t in truth if t["split"] == "main"]
    by: dict = {}
    for t in main:
        by.setdefault(tuple(t["stratum"]), []).append(t["case_id"])
    rng = random.Random(SUBSET_SEED)
    keys = sorted(by, key=str)
    quota = {k: max(4, round(SUBSET_N * len(by[k]) / len(main))) for k in keys}
    while sum(quota.values()) > SUBSET_N:
        k = max(keys, key=lambda x: quota[x])
        quota[k] -= 1
    out = []
    for k in keys:
        ids = sorted(by[k])
        rng.shuffle(ids)
        out += ids[: quota[k]]
    return sorted(out)


def spent() -> float:
    tot = 0.0
    for f in OUT.glob("*.jsonl"):
        for line in f.read_text().splitlines():
            tot += json.loads(line).get("cost_usd", 0.0)
    return tot


def run(jobs: list[tuple[str, str, str, str, list]], cap: float, workers: int, token: str) -> None:
    """jobs: (judge, view, variant, tag, cases). variant in {v1, v1r, v2}; tag names the file."""
    OUT.mkdir(exist_ok=True)
    lock = threading.Lock()
    budget = {"spent": spent(), "stopped": False}
    print(f"J2 spent so far ${budget['spent']:.3f}; cap ${cap}", flush=True)
    for judge, view, variant, tag, cases in jobs:
        model, pin, pout = J2_JUDGES[judge][:3]
        out = OUT / f"{judge}__{view}__{variant}__{tag}.jsonl"
        done = {json.loads(l)["case_id"] for l in out.read_text().splitlines()} if out.exists() else set()
        todo = [c for c in cases if c["case_id"] not in done]

        def work(c: dict) -> None:
            with lock:
                if budget["spent"] >= cap:
                    budget["stopped"] = True
                    return
            if variant == "v1r":
                prompt = build_prompt_reversed(c["views"][view], c["case_id"], judge)
            else:
                prompt = build_prompt(c["views"][view], c["case_id"], judge, variant)
            resp, dt = call(model, prompt, token)
            u = resp.get("usage") or {}
            cost = u.get("prompt_tokens", 0) * pin / 1e6 + u.get("completion_tokens", 0) * pout / 1e6
            msg = (resp.get("choices") or [{}])[0].get("message", {}) if "choices" in resp else {}
            text = msg.get("content") or ""
            row = {"case_id": c["case_id"], "judge": judge, "model": model, "view": view, "variant": variant,
                   "tag": tag, "served_model": resp.get("model"), "parsed": parse(text), "raw": text[:4000],
                   "error": resp.get("error"), "latency_s": round(dt, 2), "prompt_tokens": u.get("prompt_tokens"),
                   "completion_tokens": u.get("completion_tokens"), "cost_usd": cost,
                   "time": time.strftime("%Y-%m-%dT%H:%M:%S")}
            with lock:
                budget["spent"] += cost
                with out.open("a") as fh:
                    fh.write(json.dumps(row, ensure_ascii=False) + "\n")

        with ThreadPoolExecutor(workers) as ex:
            list(ex.map(work, todo))
        rows = [json.loads(l) for l in out.read_text().splitlines()] if out.exists() else []
        print(f"{out.name}: {len(rows)} rows, {sum(r['parsed'] is not None for r in rows)} parsed, "
              f"errors {sum(bool(r['error']) for r in rows)}; J2 total ${budget['spent']:.3f}", flush=True)
        if budget["stopped"]:
            print("HARD STOP: cap reached", flush=True)
            return


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", choices=["pilot", "aya", "subset"], required=True)
    ap.add_argument("--cap", type=float, default=4.50)
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    token = os.environ.get("HF_TOKEN") or Path(os.path.expanduser(os.environ.get(
        "HF_TOKEN_PATH", "/scratch/pabitra/.cache/huggingface/token"))).read_text().strip()
    cases = {c["case_id"]: c for c in map(json.loads, (HERE / "data/cases.jsonl").read_text().splitlines())}
    cal = [c for c in cases.values() if c["split"] == "calibration"]
    main_ = [c for c in cases.values() if c["split"] == "main"]
    sub = [cases[i] for i in subset_ids()]
    if a.run == "pilot":  # 3 calibration cases: provider responds and the reply parses; not analysed
        jobs = [("aya", "transcript", "v1", "pilot", cal[:3])]
    elif a.run == "aya":
        jobs = [("aya", v, "v1", s, cs) for s, cs in (("calibration", cal), ("main", main_))
                for v in ("transcript_state", "transcript")]
    else:
        jobs = []
        for j in ("gemma", "qwen", "gptoss", "deepseek", "aya"):
            jobs += [(j, "transcript_state", "v1r", "subset", sub), (j, "transcript_state", "v1", "subset_rep", sub)]
        jobs += [("aya", "transcript", "v2", "subset", sub)]
    run(jobs, a.cap, a.workers, token)


if __name__ == "__main__":
    main()
