"""TXN v1 model evaluation (docs/txn/TXN_SPEC.md). Reuses scripts/p1_eval.py's episode loop (`run`) and summary
(`summarise`) by importing them; only the env factory changes: `actionproof.stage3.env.make_env_s3` is patched to
`actionproof.txn.env.make_env_txn` for the duration of the run. p1_eval imports the factory inside `run`, so the patch
takes effect there. make_env_txn builds a TxnSurfaceEnv for the ood_alias and ood_schema groups. Prompts, tool
schemas, parser, tool-result format, 16 turns and 256 new tokens are p1_eval's. No reconciliation line.

vLLM (Slurm):
    python scripts/txn_eval.py --model runs/rft_proc/merged --adapter runs/X/final --tag D_s0 --split txn_test --samples 2 [--fp8]
OpenAI-compatible endpoint (greedy only; same prompts, tools, turn cap and verifier):
    python scripts/txn_eval.py --api-base https://host/v1 --api-model name --tag api_x --split txn_test \
        --price-in 0.15 --price-out 0.60 --max-usd 5     # prices in USD per 1M tokens; key from $OPENAI_API_KEY

Output: results/txn/<tag>__<split>.jsonl (one record per episode, with `txn` flags, `invariants`, `decision_record`),
plus .summary.json (p1_eval's summary and the TXN metrics with numerators and denominators). The test split is run
once per tag: an existing output file is never overwritten.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import threading
import time
import types
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
_spec = importlib.util.spec_from_file_location("p1_eval", ROOT / "scripts/p1_eval.py")
P1 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(P1)

SPLITS = ("txn_dev", "txn_val", "txn_test")


# ---------------------------------------------------------------------- env factory patch
def run_txn(llm, tasks, samples: int) -> list[dict]:
    import actionproof.stage3.env as S3
    from actionproof.txn.env import make_env_txn
    orig = S3.make_env_s3
    S3.make_env_s3 = lambda task, rid, seed=None: make_env_txn(task, rid, seed=seed)
    try:
        rows = P1.run(llm, tasks, samples, recon=False)
    finally:
        S3.make_env_s3 = orig
    return [annotate(r) for r in rows]


def annotate(r: dict) -> dict:
    from actionproof.txn import invariants, truth
    r["txn"] = truth.txn_score(r)
    r["invariants"] = invariants.check(r)
    r["decision_record"] = truth.decision_record(r)
    return r


def txn_summary(rows: list[dict]) -> dict:
    from actionproof.txn.truth import breakdown, summarise
    g = [r["txn"] for r in rows if r.get("sample", -1) < 0]
    s = [r["txn"] for r in rows if r.get("sample", -1) >= 0]
    al = [r["txn"] for r in rows]
    inv: dict[str, int] = {}
    for r in rows:
        for v in r.get("invariants") or []:
            inv[v["code"]] = inv.get(v["code"], 0) + 1
    return {"all": summarise(al), "greedy": summarise(g), "sampled": summarise(s) if s else None,
            "by_group": breakdown(al, "group"), "by_class": breakdown(al, "fault_class"),
            "by_regime": breakdown(al, "regime"), "invariant_violations": inv,
            "episodes_with_violation": sum(bool(r.get("invariants")) for r in rows)}


# ---------------------------------------------------------------------- OpenAI-compatible client (greedy only)
class _Tok:
    """Length guard only: ~3 characters per token (conservative for English JSON)."""

    def apply_chat_template(self, conv, add_generation_prompt=True, tokenize=False):
        return "\n".join(f"{m['role']}: {m['content']}" for m in conv)

    def encode(self, text, add_special_tokens=False):
        return range(len(text) // 3)


class BudgetStop(Exception):
    pass


class ApiLLM:
    """Duck-types the slice of vllm.LLM that p1_eval.run uses (get_tokenizer, llm_engine.model_config.max_model_len,
    chat). Every request is greedy (temperature 0). `max_usd` is a hard stop: a request is sent only if the cost
    spent so far, plus the worst-case cost of every request in flight, plus this request's worst case (prompt
    characters + 8 per message, as tokens, x price_in + max_tokens x price_out), stays within the cap. Once a request is refused, every later
    request returns an empty completion, which ends the episode (`ended_by = no_call`). The summary marks
    `budget_stopped`."""

    def __init__(self, base: str, model: str, key: str | None, price_in: float, price_out: float, max_usd: float,
                 ctx: int, concurrency: int, timeout: float = 120.0) -> None:
        self.base, self.model, self.key = base.rstrip("/"), model, key
        self.pin, self.pout, self.max_usd = price_in / 1e6, price_out / 1e6, max_usd
        self.llm_engine = types.SimpleNamespace(model_config=types.SimpleNamespace(max_model_len=ctx))
        self.concurrency, self.timeout = concurrency, timeout
        self.lock = threading.Lock()
        self.spent = self.reserved = 0.0
        self.tokens = {"prompt": 0, "completion": 0}
        self.requests = self.errors = self.refused = 0
        self.stopped = False

    def get_tokenizer(self):
        return _Tok()

    def _post(self, body: dict) -> dict:
        req = urllib.request.Request(f"{self.base}/chat/completions", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json",
                                              **({"Authorization": f"Bearer {self.key}"} if self.key else {})})
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            return json.loads(resp.read().decode())

    def _one(self, conv: list[dict], max_tokens: int) -> str:
        # worst case: no tokenizer emits more tokens than characters, plus a per-message overhead
        est = (len(json.dumps(conv)) + 8 * len(conv)) * self.pin + max_tokens * self.pout
        with self.lock:
            if self.stopped or self.spent + self.reserved + est > self.max_usd:
                self.stopped = True
                self.refused += 1
                return ""
            self.reserved += est
        cost, text = 0.0, ""
        try:
            for attempt in range(4):
                try:
                    out = self._post({"model": self.model, "messages": conv, "temperature": 0, "max_tokens": max_tokens})
                    u = out.get("usage") or {}
                    pt, ct = int(u.get("prompt_tokens") or 0), int(u.get("completion_tokens") or 0)
                    if not u:  # no usage reported: charge the worst case
                        cost = est
                    else:
                        cost = pt * self.pin + ct * self.pout
                    with self.lock:
                        self.tokens["prompt"] += pt
                        self.tokens["completion"] += ct
                        self.requests += 1
                    text = (out.get("choices") or [{}])[0].get("message", {}).get("content") or ""
                    break
                except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, json.JSONDecodeError) as e:
                    code = getattr(e, "code", None)
                    if code is not None and code < 500 and code != 429:
                        raise
                    time.sleep(2 ** attempt)
            else:
                with self.lock:
                    self.errors += 1
        finally:
            with self.lock:
                self.reserved -= est
                self.spent += cost
        return text

    def chat(self, convs, params, use_tqdm=False):
        for p in params:
            if getattr(p, "temperature", 0) not in (0, 0.0):
                raise SystemExit("the API path is greedy only (run with --samples 0)")
        with ThreadPoolExecutor(self.concurrency) as ex:
            texts = list(ex.map(lambda cp: self._one(cp[0], getattr(cp[1], "max_tokens", 256)), zip(convs, params)))
        return [types.SimpleNamespace(outputs=[types.SimpleNamespace(text=t)]) for t in texts]

    def report(self) -> dict:
        return {"usd_spent": round(self.spent, 6), "max_usd": self.max_usd, "budget_stopped": self.stopped,
                "requests": self.requests, "refused": self.refused, "errors": self.errors, "tokens": self.tokens}


def _ensure_sampling_params() -> None:
    try:
        import vllm  # noqa: F401
    except ImportError:  # API path on a machine without vLLM: p1_eval.run only builds SamplingParams objects
        mod = types.ModuleType("vllm")

        class SamplingParams:
            def __init__(self, temperature=1.0, max_tokens=16, seed=None, **kw):
                self.temperature, self.max_tokens, self.seed = temperature, max_tokens, seed
        mod.SamplingParams = SamplingParams
        sys.modules["vllm"] = mod


# ---------------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model")
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--split", choices=SPLITS, required=True)
    ap.add_argument("--samples", type=int, default=2)
    ap.add_argument("--fp8", action="store_true")
    ap.add_argument("--gpu-mem", type=float, default=0.85)
    ap.add_argument("--n-tasks", type=int, default=None)
    ap.add_argument("--api-base", default=None)
    ap.add_argument("--api-model", default=None)
    ap.add_argument("--api-key-env", default="OPENAI_API_KEY")
    ap.add_argument("--price-in", type=float, default=None, help="USD per 1M prompt tokens")
    ap.add_argument("--price-out", type=float, default=None, help="USD per 1M completion tokens")
    ap.add_argument("--max-usd", type=float, default=None)
    ap.add_argument("--api-ctx", type=int, default=32768)
    ap.add_argument("--concurrency", type=int, default=8)
    a = ap.parse_args()
    api = a.api_base is not None
    if api and (a.api_model is None or a.price_in is None or a.price_out is None or a.max_usd is None):
        raise SystemExit("--api-base needs --api-model, --price-in, --price-out and --max-usd")
    if not api and not a.model:
        raise SystemExit("--model is required without --api-base")
    out = ROOT / f"results/txn/{a.tag}__{a.split}.jsonl"
    if out.exists():
        raise SystemExit(f"{out} exists; each tag and split is run once")
    from actionproof.bench.generator_proc import DATA, load_tasks
    tasks = load_tasks(DATA / f"{a.split}.jsonl")[: a.n_tasks]
    if a.split == "txn_test" and a.n_tasks is not None:
        raise SystemExit("the locked test split is run in full")
    t0 = time.time()
    if api:
        _ensure_sampling_params()
        llm = ApiLLM(a.api_base, a.api_model, os.environ.get(a.api_key_env), a.price_in, a.price_out, a.max_usd,
                     a.api_ctx, a.concurrency)
        samples = 0
    else:
        from vllm import LLM
        from actionproof.train.merge import merge
        model = merge(a.adapter, a.model) if a.adapter else a.model
        kw = {"quantization": "fp8"} if a.fp8 else {}
        llm = LLM(model=model, max_model_len=8192, gpu_memory_utilization=a.gpu_mem, seed=0, enable_prefix_caching=True,
                  **kw)
        samples = a.samples
    rows = run_txn(llm, tasks, samples)
    for r in rows:
        r["tag"] = a.tag
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(json.dumps(r, default=str) + "\n" for r in rows))
    s = {"tag": a.tag, "split": a.split, "model": a.model or a.api_model, "adapter": a.adapter, "fp8": a.fp8,
         "api_base": a.api_base, "samples": samples, "n_tasks": len(tasks), "seconds": round(time.time() - t0, 1),
         **P1.summarise(rows), "txn": txn_summary(rows)}
    if api:
        s["api"] = llm.report()
    out.with_suffix(".summary.json").write_text(json.dumps(s, indent=2))
    print(json.dumps({"tag": a.tag, "split": a.split, "STR": s["txn"]["all"]["STR"], "VTC": s["txn"]["all"]["VTC"],
                      **({"api": s["api"]} if api else {})}))


if __name__ == "__main__":
    main()
