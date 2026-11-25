"""B1 addendum 1 (PREREGISTRATION_B1_ADDENDUM1.md): open frontier models through the HF Inference Providers router,
as prompted agents on the frozen P1 test, with P1's episode loop unchanged.

`ApiLLM` stands in for the vLLM engine that `scripts/p1_eval.py::run` expects: `chat(convs, params)` sends each
conversation (system prompt, task, tool results: exactly what a local model sees) to the router at temperature 0,
and `get_tokenizer()` returns an approximate tokenizer (3.5 characters per token) so that P1's 8,192-token context
guard applies to the API models too. Greedy only (one run per task).

Budget: every call's cost is the router-listed price per token (`evaluation/llm_judge_committee/committee.py`)
times the returned `usage`. It is appended to `results/b1/api_cost.jsonl`, and a call is refused once the
cumulative total over that file would exceed `--cap` (default $10, the author's limit for B1).

    python scripts/b1_api_eval.py --model qwen --split p1_test_full
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("p1_eval", ROOT / "scripts/p1_eval.py")
P1 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(P1)
URL = "https://router.huggingface.co/v1/chat/completions"
# tag: (router model id with pinned provider, $ per 1M input, $ per 1M output, max new tokens, extra body)
MODELS = {
    "qwen": ("Qwen/Qwen3-235B-A22B-Instruct-2507:deepinfra", 0.09, 0.55, 256, {}),
    "gptoss": ("openai/gpt-oss-120b:baseten", 0.10, 0.50, 1024, {"reasoning_effort": "low"}),
}
COST = ROOT / "results/b1/api_cost.jsonl"
LOCK = threading.Lock()


def spent() -> float:
    return sum(json.loads(l)["cost_usd"] for l in COST.read_text().splitlines()) if COST.exists() else 0.0


class _Out:
    def __init__(self, text: str):
        self.outputs = [type("O", (), {"text": text})()]


class _Tok:
    def apply_chat_template(self, conv, add_generation_prompt=True, tokenize=False):
        return "\n".join(m["content"] for m in conv)

    def encode(self, text, add_special_tokens=False):
        return range(int(len(text) / 3.5))


class ApiLLM:
    def __init__(self, tag: str, token: str, cap: float, split: str):
        self.tag, self.token, self.cap, self.split = tag, token, cap, split
        self.model, self.pin, self.pout, self.max_tokens, self.extra = MODELS[tag]
        self.llm_engine = type("E", (), {"model_config": type("C", (), {"max_model_len": 8192})()})()

    def get_tokenizer(self):
        return _Tok()

    def _one(self, conv) -> str:
        with LOCK:
            if spent() >= self.cap:
                raise SystemExit(f"B1 API hard stop: ${spent():.2f} >= ${self.cap}")
        body = json.dumps({"model": self.model, "messages": conv, "temperature": 0, "max_tokens": self.max_tokens,
                           **self.extra}).encode()
        for k in range(5):
            t0 = time.time()
            try:
                req = urllib.request.Request(URL, data=body, headers={"Authorization": f"Bearer {self.token}",
                                                                      "Content-Type": "application/json"})
                with urllib.request.urlopen(req, timeout=180) as r:
                    resp = json.loads(r.read())
                break
            except urllib.error.HTTPError as e:
                if e.code in (400, 401, 403, 404, 422) or k == 4:
                    resp = {"error": f"HTTP {e.code}: {e.read().decode()[:200]}"}
                    break
            except Exception as e:  # network errors and timeouts are retried
                if k == 4:
                    resp = {"error": repr(e)[:200]}
                    break
            time.sleep(2 ** k)
        u = resp.get("usage") or {}
        cost = (u.get("prompt_tokens", 0) * self.pin + u.get("completion_tokens", 0) * self.pout) / 1e6
        with LOCK:
            with COST.open("a") as f:
                f.write(json.dumps({"tag": self.tag, "split": self.split, "model": self.model, "cost_usd": cost,
                                    "prompt_tokens": u.get("prompt_tokens"), "completion_tokens": u.get("completion_tokens"),
                                    "latency_s": round(time.time() - t0, 2), "error": resp.get("error")}) + "\n")
        if "error" in resp:
            return ""
        return (resp["choices"][0]["message"].get("content") or "").strip()

    def chat(self, convs, params, use_tqdm=False):
        with ThreadPoolExecutor(8) as ex:
            return [_Out(t) for t in ex.map(self._one, convs)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=sorted(MODELS))
    ap.add_argument("--split", required=True)
    ap.add_argument("--cap", type=float, default=10.0)
    a = ap.parse_args()
    if not a.split.startswith("p1_test_"):
        raise SystemExit("B1 runs only the frozen P1 test splits")
    out = ROOT / f"results/b1/{a.model}__{a.split}.jsonl"
    if out.exists():
        raise SystemExit(f"{out} exists; each model and split runs once")
    from actionproof.bench.generator_proc import DATA, load_tasks
    token = os.environ.get("HF_TOKEN") or Path(os.path.expanduser(os.environ.get(
        "HF_TOKEN_PATH", "/scratch/pabitra/.cache/huggingface/token"))).read_text().strip()
    COST.parent.mkdir(parents=True, exist_ok=True)
    tasks = load_tasks(DATA / f"{a.split}.jsonl")
    t0 = time.time()
    rows = P1.run(ApiLLM(a.model, token, a.cap, a.split), tasks, 0, False)
    out.write_text("".join(json.dumps(r, default=str) + "\n" for r in rows))
    s = {"tag": a.model, "split": a.split, "phase": "b1_api", "model": MODELS[a.model][0], "samples": 0,
         "seconds": round(time.time() - t0, 1), "cost_usd_total_so_far": spent(), **P1.summarise(rows)}
    out.with_suffix(".summary.json").write_text(json.dumps(s, indent=2))
    print(json.dumps({"tag": a.model, "split": a.split, "greedy_vsc": s["greedy"]["vsc"], "spent": round(spent(), 3)}))


if __name__ == "__main__":
    main()
