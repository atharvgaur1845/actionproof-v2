"""R1 capability retention: score generations with the official checkers (PREREGISTRATION_R1.md §5).

Runs in `.venv-r1`. For each `results/r1/gen/<tag>.jsonl` writes `results/r1/score/<tag>.jsonl` (one row per
input) and `<tag>.summary.json`.

- IFEval: lm-eval 0.4.13 `process_results` (strict and loose; prompt and instruction level).
- BFCL v4 (bfcl-eval 2026.3.23), following its `eval_runner`:
  - AST categories: `default_decode_ast_prompting`, then `is_function_calling_format_output`, then
    `ast_checker` against the possible answers. Decoding failures are counted separately as format failures.
  - irrelevance and live_irrelevance: correct when no function call decodes (`is_empty_output`).
  - live_relevance: correct when one does.
  - The checker's model key is a prompting-mode entry with `underscore_to_dot = False`, so function names
    with "." are compared unchanged.
- Over-length inputs (`over_length`) are dropped; the set is the same for every model.

    .venv-r1/bin/python scripts/r1_score.py [tag ...]      # default: every generation file without a score
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GEN, SCORE = ROOT / "results/r1/gen", ROOT / "results/r1/score"
CHECKER_MODEL = "Qwen_Qwen3-8B"  # escaped form; resolves to MODEL_CONFIG_MAPPING["Qwen/Qwen3-8B"]
NONLIVE_AST = ("simple_python", "multiple", "parallel", "parallel_multiple")
LIVE_AST = ("live_simple", "live_multiple", "live_parallel", "live_parallel_multiple")
IRRELEVANCE = ("irrelevance", "live_irrelevance")


def _bfcl_tables() -> tuple[dict, dict]:
    import bfcl_eval
    d = Path(bfcl_eval.__file__).parent / "data"
    prompts, answers = {}, {}
    for cat in NONLIVE_AST + LIVE_AST + IRRELEVANCE + ("live_relevance",):
        for line in (d / f"BFCL_v4_{cat}.json").read_text().splitlines():
            e = json.loads(line)
            prompts[e["id"]] = e
        pa = d / "possible_answer" / f"BFCL_v4_{cat}.json"
        if pa.exists():
            for line in pa.read_text().splitlines():
                e = json.loads(line)
                answers[e["id"]] = e["ground_truth"]
    return prompts, answers


def score_bfcl(row: dict, prompts: dict, answers: dict) -> dict:
    from bfcl_eval.constants.enums import Language, ReturnFormat
    from bfcl_eval.eval_checker.ast_eval.ast_checker import ast_checker
    from bfcl_eval.model_handler.utils import default_decode_ast_prompting
    from bfcl_eval.utils import is_empty_output, is_function_calling_format_output

    cat, text = row["category"], row["text"]
    try:
        decoded = default_decode_ast_prompting(text, ReturnFormat.PYTHON, False)
        decode_ok = True
    except Exception:  # the official runner treats any decoding exception as "no valid call"
        decoded, decode_ok = None, False
    if cat in IRRELEVANCE or cat == "live_relevance":
        has_call = decode_ok and not is_empty_output(decoded)
        correct = (not has_call) if cat in IRRELEVANCE else has_call
        return {"correct": bool(correct), "format_ok": decode_ok, "has_call": bool(has_call)}
    if not decode_ok or not is_function_calling_format_output(decoded):
        return {"correct": False, "format_ok": False, "has_call": False}
    res = ast_checker(prompts[row["id"]]["function"], decoded, answers[row["id"]], Language.PYTHON, cat, CHECKER_MODEL)
    return {"correct": bool(res["valid"]), "format_ok": True, "has_call": len(decoded) > 0,
            "error_type": res.get("error_type")}


def score_ifeval(row: dict, docs: dict) -> dict:
    from lm_eval.tasks.ifeval.utils import process_results
    r = process_results(docs[row["id"]], [row["text"]])
    return {"correct": bool(r["prompt_level_strict_acc"]), "prompt_strict": bool(r["prompt_level_strict_acc"]),
            "prompt_loose": bool(r["prompt_level_loose_acc"]), "inst_strict": r["inst_level_strict_acc"],
            "inst_loose": r["inst_level_loose_acc"]}


def summarise(rows: list[dict]) -> dict:
    def acc(cats):
        xs = [r["correct"] for r in rows if r["category"] in cats]
        return (sum(xs) / len(xs)) if xs else None, len(xs)
    s = {"n_scored": len(rows)}
    for c in sorted({r["category"] for r in rows}):
        s[c] = dict(zip(("acc", "n"), acc((c,))))
    # BFCL's own aggregation: unweighted mean of the non-live AST categories; live AST weighted by count
    nl = [s[c]["acc"] for c in NONLIVE_AST if c in s]
    s["bfcl_nonlive_ast"] = sum(nl) / len(nl) if len(nl) == len(NONLIVE_AST) else None
    s["bfcl_live_ast"] = acc(LIVE_AST)[0]
    s["bfcl_irrelevance"] = acc(("irrelevance",))[0]
    s["bfcl_live_irrelevance"] = acc(("live_irrelevance",))[0]
    b = [r for r in rows if r["bench"] == "bfcl" and r["category"] not in IRRELEVANCE + ("live_relevance",)]
    s["bfcl_format_fail_rate"] = (sum(not r["format_ok"] for r in b) / len(b)) if b else None
    ie = [r for r in rows if r["bench"] == "ifeval"]
    if ie:
        s["ifeval_prompt_strict"] = sum(r["prompt_strict"] for r in ie) / len(ie)
        s["ifeval_prompt_loose"] = sum(r["prompt_loose"] for r in ie) / len(ie)
        inst = [x for r in ie for x in r["inst_strict"]]
        s["ifeval_inst_strict"] = sum(inst) / len(inst)
        insl = [x for r in ie for x in r["inst_loose"]]
        s["ifeval_inst_loose"] = sum(insl) / len(insl)
    return s


def main() -> None:
    SCORE.mkdir(parents=True, exist_ok=True)
    tags = sys.argv[1:] or sorted(p.name[:-6] for p in GEN.glob("*.jsonl") if not (SCORE / p.name).exists())
    prompts, answers = _bfcl_tables()
    docs = {str(json.loads(line)["key"]): json.loads(line)
            for line in (ROOT / "data/retention/ifeval_input_data.jsonl").read_text().splitlines()}
    for tag in tags:
        rows = []
        gen = [json.loads(line) for line in (GEN / f"{tag}.jsonl").read_text().splitlines()]
        for g in gen:
            if g["over_length"]:
                continue
            sc = score_ifeval(g, docs) if g["bench"] == "ifeval" else score_bfcl(g, prompts, answers)
            rows.append({"bench": g["bench"], "id": g["id"], "category": g["category"],
                         "output_tokens": g["output_tokens"], **sc})
        (SCORE / f"{tag}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
        s = {"tag": tag, "over_length_dropped": sum(g["over_length"] for g in gen), **summarise(rows)}
        (SCORE / f"{tag}.summary.json").write_text(json.dumps(s, indent=2))
        print(tag, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in s.items()
                    if k.startswith(("bfcl_", "ifeval_", "over"))})


if __name__ == "__main__":
    main()
