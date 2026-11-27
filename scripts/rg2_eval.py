"""RG2 model evaluation (PREREGISTRATION_RG2.md): the TXN/SEL1 episode loop (scripts/p1_eval.py::run via
scripts/txn_eval.py), SEL1's env (transient outages) with gate v1, gate v2 or no gate in front
(actionproof/recovery_gate/rg2_env.py).

    python scripts/rg2_eval.py --model M [--adapter A] --tag V2_D_s0 --gate v2 --split rg2_test
Output: results/rg2/<tag>__<split>.jsonl (+ .summary.json); rg2_test runs once per tag, after scripts/freeze_rg2.py
--check.
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
    from actionproof.recovery_gate.rg2_env import make_env_rg2
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--gate", choices=("v1", "v2", "none"), required=True)
    ap.add_argument("--split", choices=("rg2_test", "sel1_train"), required=True)
    ap.add_argument("--samples", type=int, default=2)
    ap.add_argument("--n-tasks", type=int, default=None)
    a = ap.parse_args()
    out = ROOT / f"results/rg2/{a.tag}__{a.split}.jsonl"
    if out.exists():
        raise SystemExit(f"{out} exists")
    if a.split == "rg2_test":
        if a.n_tasks is not None:
            raise SystemExit("rg2_test is run in full")
        if subprocess.run([sys.executable, str(ROOT / "scripts/freeze_rg2.py"), "--check"], cwd=ROOT).returncode:
            raise SystemExit("freeze check failed")
    gate = None if a.gate == "none" else a.gate
    import actionproof.stage3.env as S3
    from actionproof.bench.generator_proc import DATA, load_tasks
    from vllm import LLM
    from actionproof.train.merge import merge
    tasks = load_tasks(DATA / f"{a.split}.jsonl")[: a.n_tasks]
    model = merge(a.adapter, a.model) if a.adapter else a.model
    llm = LLM(model=model, max_model_len=8192, gpu_memory_utilization=0.85, seed=0, enable_prefix_caching=True)
    S3E = TE.P1.E1.S3E
    orig_env, orig_rec = S3.make_env_s3, S3E.episode_record_proc

    def record(task, env):
        if getattr(env, "gate", None) is not None:
            env.gate_finalize()
        r = orig_rec(task, env)
        if getattr(env, "gate", None) is not None:
            r["gate"] = env.gate_record()
        return r
    t0 = time.time()
    S3.make_env_s3 = lambda task, rid, seed=None: make_env_rg2(task, rid, seed=seed, gate=gate)
    S3E.episode_record_proc = record
    try:
        rows = [TE.annotate(r) for r in TE.P1.run(llm, tasks, a.samples, recon=False)]
    finally:
        S3.make_env_s3, S3E.episode_record_proc = orig_env, orig_rec
    for r in rows:
        r["tag"] = a.tag
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(json.dumps(r, default=str) + "\n" for r in rows))
    s = {"tag": a.tag, "split": a.split, "gate": a.gate, "model": a.model, "adapter": a.adapter, "samples": a.samples,
         "n_tasks": len(tasks), "seconds": round(time.time() - t0, 1), "txn": TE.txn_summary(rows),
         "gate_events": RG.gate_summary(rows)}
    out.with_suffix(".summary.json").write_text(json.dumps(s, indent=2))
    print(json.dumps({"tag": a.tag, "VTC": s["txn"]["all"]["VTC"], "STR": s["txn"]["all"]["STR"]}))


if __name__ == "__main__":
    main()
