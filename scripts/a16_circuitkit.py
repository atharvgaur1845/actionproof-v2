"""A16 exploratory CircuitKIT run (PREREGISTRATION_A15 §5): EAP-IG at node level on matched dev decision points of
R_V seed 0. Clean = partial-payment branch (answer: the first token of 'refund_payment'); corrupt = the same task's
complete branch (answer: the first token of 'post_journal_entry'). Prompts carry the model's chat template plus the
tool-call stem. Never used for a claim.
    python scripts/a16_circuitkit.py --weights runs/a12_proc_rv_s0/merged --out results/a16/circuitkit
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("a16_mech", ROOT / "scripts/a16_mech.py")
m16 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(m16)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--out", default="results/a16/circuitkit")
    ap.add_argument("--n", type=int, default=20)
    a = ap.parse_args()
    import circuitkit  # noqa: F401  (must be imported before transformer_lens)
    from circuitkit import Pipeline
    from transformers import AutoTokenizer
    out = ROOT / a.out
    out.mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(ROOT / a.weights)
    exs = m16.examples("proc_mech_dev")
    rows = []
    for p, c in m16.pairs(exs)[: a.n]:
        tp = tok.apply_chat_template(p["conv"], add_generation_prompt=True, tokenize=False) + m16.STEM
        tc = tok.apply_chat_template(c["conv"], add_generation_prompt=True, tokenize=False) + m16.STEM
        rows.append({"question": tp, "answer": "refund", "corrupted": tc, "corrupt_answer": "post"})
    csv_path = out / "pairs.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    status = {"n_pairs": len(rows), "answer_tokens": {x: tok(x, add_special_tokens=False).input_ids for x in ("refund", "post")}}
    try:
        pipe = Pipeline.from_custom_data(str(ROOT / a.weights), str(csv_path), clean_prompt="{question}",
                                         clean_answer="{answer}", corrupt_prompt="{corrupted}",
                                         corrupt_answer="{corrupt_answer}", task_name="a16_partial_vs_complete",
                                         output_dir=str(out / "run"))
        res = pipe.discover(algorithm="eap-ig", level="node", sparsity=0.1, n_examples=len(rows), batch_size=1,
                            ig_steps=3)
        status["discover"] = "ok"
        status["discover_repr"] = repr(res)[:2000]
        try:
            ev = pipe.evaluate()
            status["evaluate"] = "ok"
            status["evaluate_repr"] = repr(ev)[:4000]
        except Exception as e:  # noqa: BLE001
            status["evaluate"] = f"{type(e).__name__}: {e}"
    except Exception as e:  # noqa: BLE001
        status["discover"] = f"{type(e).__name__}: {e}"
        status["traceback"] = traceback.format_exc()[-3000:]
    (out / "status.json").write_text(json.dumps(status, indent=2, default=str))
    print(json.dumps({k: v for k, v in status.items() if k != "traceback"}, indent=1, default=str)[:3000])


if __name__ == "__main__":
    main()
