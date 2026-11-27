"""RG1 addendum 1 (PREREGISTRATION_RG1_ADDENDUM1.md, descriptive): the frozen gate on SEL1's sealed test, which has
transient outages. Same loop as scripts/rg1_eval.py; env = actionproof.recovery_gate.sel_env.make_env_gated_sel.

    python scripts/rg1x_sel_eval.py --model M [--adapter A] --tag X1_RG_D_s0
Output: results/rg1/<tag>__sel1_test.jsonl (+ .summary.json); run once per tag, after scripts/freeze_rg1.py --check.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
_spec = importlib.util.spec_from_file_location("rg1_eval", ROOT / "scripts/rg1_eval.py")
RG = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(RG)
TE = RG.TE


def main() -> None:
    from actionproof.recovery_gate.gate import Rules
    from actionproof.recovery_gate.sel_env import make_env_gated_sel
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--samples", type=int, default=2)
    a = ap.parse_args()
    out = ROOT / f"results/rg1/{a.tag}__sel1_test.jsonl"
    if out.exists():
        raise SystemExit(f"{out} exists")
    if subprocess.run([sys.executable, str(ROOT / "scripts/freeze_rg1.py"), "--check"], cwd=ROOT).returncode:
        raise SystemExit("freeze check failed: the gate differs from the registration")
    import actionproof.stage3.env as S3
    from actionproof.bench.generator_proc import DATA, load_tasks
    from vllm import LLM
    from actionproof.train.merge import merge
    tasks = load_tasks(DATA / "sel1_test.jsonl")
    model = merge(a.adapter, a.model) if a.adapter else a.model
    llm = LLM(model=model, max_model_len=8192, gpu_memory_utilization=0.85, seed=0, enable_prefix_caching=True)
    S3E = TE.P1.E1.S3E
    orig_env, orig_rec = S3.make_env_s3, S3E.episode_record_proc

    def record(task, env):
        env.gate_finalize()
        r = orig_rec(task, env)
        r["gate"] = env.gate_record()
        return r
    t0 = time.time()
    S3.make_env_s3 = lambda task, rid, seed=None: make_env_gated_sel(task, rid, seed=seed, rules=Rules())
    S3E.episode_record_proc = record
    try:
        rows = [TE.annotate(r) for r in TE.P1.run(llm, tasks, a.samples, recon=False)]
    finally:
        S3.make_env_s3, S3E.episode_record_proc = orig_env, orig_rec
    for r in rows:
        r["tag"] = a.tag
    out.write_text("".join(json.dumps(r, default=str) + "\n" for r in rows))
    s = {"tag": a.tag, "split": "sel1_test", "rules": "full", "model": a.model, "adapter": a.adapter,
         "samples": a.samples, "n_tasks": len(tasks), "seconds": round(time.time() - t0, 1),
         "code_sha256": RG.code_hashes(), "txn": TE.txn_summary(rows), "gate": RG.gate_summary(rows)}
    out.with_suffix(".summary.json").write_text(json.dumps(s, indent=2))
    print(json.dumps({"tag": a.tag, "VTC": s["txn"]["all"]["VTC"], "STR": s["txn"]["all"]["STR"]}))


if __name__ == "__main__":
    main()
