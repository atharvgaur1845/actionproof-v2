"""R1 capability retention: build the frozen chat inputs for IFEval and BFCL (PREREGISTRATION_R1.md §3).

Runs in the scoring venv (`.venv-r1`, `configs/r1_scoring_requirements.txt`), which holds the official
checkers: bfcl-eval 2026.3.23 (BFCL v4 data) and lm-eval 0.4.13 (IFEval). The cluster only applies the
model's chat template and generates, so the exact inputs every model sees are fixed and hashed here.

BFCL inputs are built by BFCL's own `system_prompt_pre_processing_chat_model` (its prompting-mode
default: python call format, json function docs, plaintext, classic style), as its OSS handler does.
IFEval inputs are the single user turn, as in lm-eval.

    .venv-r1/bin/python -m actionproof.retention.build_prompts   # writes data/retention/r1_inputs.jsonl
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data/retention/r1_inputs.jsonl"
IFEVAL = ROOT / "data/retention/ifeval_input_data.jsonl"

# Single-turn BFCL v4 categories used by R1. Multi-turn, memory, web search, format sensitivity and the
# Java/JavaScript categories are excluded for the reasons given in PREREGISTRATION_R1.md §3.
BFCL_CATEGORIES = ("simple_python", "multiple", "parallel", "parallel_multiple", "irrelevance",
                   "live_simple", "live_multiple", "live_parallel", "live_parallel_multiple",
                   "live_irrelevance", "live_relevance")

# Generation budget per benchmark (lm-eval's IFEval default is 1280; BFCL calls are short).
MAX_TOKENS = {"ifeval": 1280, "bfcl": 512}


def bfcl_dir() -> Path:
    import bfcl_eval
    return Path(bfcl_eval.__file__).parent / "data"


def bfcl_items() -> list[dict]:
    from bfcl_eval.model_handler.utils import system_prompt_pre_processing_chat_model
    items = []
    for cat in BFCL_CATEGORIES:
        for line in (bfcl_dir() / f"BFCL_v4_{cat}.json").read_text().splitlines():
            e = json.loads(line)
            turns = copy.deepcopy(e["question"])
            assert len(turns) == 1, f"{e['id']} is multi-turn"
            msgs = system_prompt_pre_processing_chat_model(turns[0], e["function"], e["id"])
            items.append({"bench": "bfcl", "id": e["id"], "category": cat, "messages": msgs,
                          "max_tokens": MAX_TOKENS["bfcl"]})
    return items


def ifeval_items() -> list[dict]:
    items = []
    for line in IFEVAL.read_text().splitlines():
        e = json.loads(line)
        items.append({"bench": "ifeval", "id": str(e["key"]), "category": "ifeval",
                      "messages": [{"role": "user", "content": e["prompt"]}],
                      "max_tokens": MAX_TOKENS["ifeval"]})
    return items


def main() -> None:
    items = ifeval_items() + bfcl_items()
    ids = [(i["bench"], i["id"]) for i in items]
    assert len(ids) == len(set(ids)), "duplicate ids"
    OUT.write_text("".join(json.dumps(i, ensure_ascii=False) + "\n" for i in items))
    by = {}
    for i in items:
        by[i["category"]] = by.get(i["category"], 0) + 1
    print(f"wrote {len(items)} inputs to {OUT.relative_to(ROOT)}: {by}")


if __name__ == "__main__":
    main()
