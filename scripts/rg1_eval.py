"""RG1 model evaluation (PREREGISTRATION_RG1.md): the TXN v1 episode loop of scripts/txn_eval.py (p1_eval's `run`:
same prompts, tool schemas, parser, 16 turns, 256 new tokens, verifier), with the Recovery Gate in front of the tools.

Only two things are patched for the duration of the run:
- the env factory (`actionproof.stage3.env.make_env_s3`) builds `make_env_gated(task, ..., rules)`;
- the episode-record builder first lets the gate finalise an episode that ended without a report, then attaches the
  gate's log (`rec["gate"]`).
`--rules none` runs the same loop with no gate (the no-gate arms on rg1_test go through this script too).

    python scripts/rg1_eval.py --model runs/p1_dwarm/merged --adapter runs/p1_d_s0/final --tag RG_D_s0 \
        --split rg1_test --rules full --samples 2
Output: results/rg1/<tag>__<split>.jsonl (+ .summary.json). rg1_test and txn_test run once per tag, in full, and only
after `python scripts/freeze_rg1.py --check` passes.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
_spec = importlib.util.spec_from_file_location("txn_eval", ROOT / "scripts/txn_eval.py")
TE = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(TE)

SPLITS = ("txn_dev", "txn_val", "txn_test", "rg1_test")
SEALED = ("txn_test", "rg1_test")
CODE = ("actionproof/recovery_gate/gate.py", "actionproof/recovery_gate/env.py", "scripts/rg1_eval.py")


def code_hashes() -> dict[str, str]:
    return {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in CODE}


def run_gated(llm, tasks, samples: int, rules) -> list[dict]:
    import actionproof.stage3.env as S3
    from actionproof.recovery_gate.env import make_env_gated
    S3E = TE.P1.E1.S3E
    orig_env, orig_rec = S3.make_env_s3, S3E.episode_record_proc

    def record(task, env):
        if hasattr(env, "gate_finalize"):
            env.gate_finalize()
        r = orig_rec(task, env)
        if getattr(env, "gate", None) is not None:
            r["gate"] = env.gate_record()
        return r

    S3.make_env_s3 = lambda task, rid, seed=None: make_env_gated(task, rid, seed=seed, rules=rules)
    S3E.episode_record_proc = record
    try:
        rows = TE.P1.run(llm, tasks, samples, recon=False)
    finally:
        S3.make_env_s3, S3E.episode_record_proc = orig_env, orig_rec
    return [TE.annotate(r) for r in rows]


def gate_summary(rows: list[dict]) -> dict:
    from collections import Counter
    c: Counter = Counter()
    eps: Counter = Counter()
    for r in rows:
        g = r.get("gate") or {}
        seen = set()
        for e in g.get("log") or []:
            k = f"{e['rule']}:{e['action']}"
            c[k] += 1
            seen.add(k)
        for k in seen:
            eps[k] += 1
    n = len(rows)
    return {"n": n, "events": dict(sorted(c.items())), "episodes_with_event": dict(sorted(eps.items())),
            "gate_calls_per_episode": sum((r.get("gate") or {}).get("gate_calls", 0) for r in rows) / n if n else None}


def main() -> None:
    from actionproof.recovery_gate.gate import ABLATIONS, rules_of
    ap = argparse.ArgumentParser()
    ap.add_argument("--model")
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--split", choices=SPLITS, required=True)
    ap.add_argument("--rules", choices=sorted(ABLATIONS) + ["none"], required=True)
    ap.add_argument("--samples", type=int, default=2)
    ap.add_argument("--gpu-mem", type=float, default=0.85)
    ap.add_argument("--n-tasks", type=int, default=None)
    ap.add_argument("--api-base", default=None)
    ap.add_argument("--api-model", default=None)
    ap.add_argument("--api-key-env", default="OPENAI_API_KEY")
    ap.add_argument("--price-in", type=float, default=None)
    ap.add_argument("--price-out", type=float, default=None)
    ap.add_argument("--max-usd", type=float, default=None)
    ap.add_argument("--api-ctx", type=int, default=32768)
    ap.add_argument("--concurrency", type=int, default=8)
    a = ap.parse_args()
    api = a.api_base is not None
    out = ROOT / f"results/rg1/{a.tag}__{a.split}.jsonl"
    if out.exists():
        raise SystemExit(f"{out} exists; each tag and split is run once")
    if a.split in SEALED:
        if a.n_tasks is not None:
            raise SystemExit("sealed splits are run in full")
        chk = subprocess.run([sys.executable, str(ROOT / "scripts/freeze_rg1.py"), "--check"], cwd=ROOT)
        if chk.returncode != 0:
            raise SystemExit("freeze check failed: the gate code or the sealed split differs from the registration")
    rules = None if a.rules == "none" else rules_of(a.rules)
    from actionproof.bench.generator_proc import DATA, load_tasks
    tasks = load_tasks(DATA / f"{a.split}.jsonl")[: a.n_tasks]
    t0 = time.time()
    if api:
        if a.api_model is None or a.price_in is None or a.price_out is None or a.max_usd is None:
            raise SystemExit("--api-base needs --api-model, --price-in, --price-out and --max-usd")
        TE._ensure_sampling_params()
        llm = TE.ApiLLM(a.api_base, a.api_model, os.environ.get(a.api_key_env), a.price_in, a.price_out, a.max_usd,
                        a.api_ctx, a.concurrency)
        samples = 0
    else:
        from vllm import LLM
        from actionproof.train.merge import merge
        model = merge(a.adapter, a.model) if a.adapter else a.model
        llm = LLM(model=model, max_model_len=8192, gpu_memory_utilization=a.gpu_mem, seed=0, enable_prefix_caching=True)
        samples = a.samples
    rows = run_gated(llm, tasks, samples, rules)
    for r in rows:
        r["tag"] = a.tag
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(json.dumps(r, default=str) + "\n" for r in rows))
    s = {"tag": a.tag, "split": a.split, "rules": a.rules, "model": a.model or a.api_model, "adapter": a.adapter,
         "api_base": a.api_base, "samples": samples, "n_tasks": len(tasks), "seconds": round(time.time() - t0, 1),
         "code_sha256": code_hashes(), **TE.P1.summarise(rows), "txn": TE.txn_summary(rows), "gate": gate_summary(rows)}
    if api:
        s["api"] = llm.report()
    out.with_suffix(".summary.json").write_text(json.dumps(s, indent=2))
    print(json.dumps({"tag": a.tag, "split": a.split, "rules": a.rules, "VTC": s["txn"]["all"]["VTC"],
                      "FCR": s["txn"]["all"]["FCR"], "STR": s["txn"]["all"]["STR"]}))


if __name__ == "__main__":
    main()
