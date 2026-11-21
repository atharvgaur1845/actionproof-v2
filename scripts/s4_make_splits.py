"""Stage 4 shift splits (docs/research/BENCHMARK_S4.md). Fresh seeds; all ids disjoint from every earlier split.

  s4_lang_en, s4_lang_hi, s4_lang_hinglish, s4_lang_para   the SAME 150 tasks (seed 302), request text varied only
  s4_api                                                   150 tasks (seed 301), evaluated in ApiShiftEnv
  s4_wf4                                                   100 tasks (seed 303), unseen order-receive-pay-post chain
  s4_combo                                                 100 tasks (seed 304), two faults on two steps
Fault mix for single-fault splits: clean .15, phantom .20, partial .20 (ratio 1/4, 1/2, 3/4), lost_ack .10,
precommit_timeout .10, never_commit .10, dup_delivery (pay) .15 -- the S3 confirmatory mix.
    python scripts/s4_make_splits.py
"""
from __future__ import annotations

import copy
import random

import actionproof.bench.generator_proc as G
from actionproof.stage3 import tasks as T
from actionproof.stage4.shift import apply_fault, combo_task, prompt_variant, workflow4_task

MIX = (("clean", 0.15), ("phantom", 0.20), ("partial", 0.20), ("lost_ack", 0.10), ("precommit_timeout", 0.10),
       ("never_commit", 0.10), ("dup_delivery", 0.15))


def single(split: str, n: int, seed: int, make=G.make_task) -> list:
    rng = random.Random(seed)
    out = []
    for i in range(n):
        t = make(split, i, rng)
        t.meta.update(faults=[], never_commit=None)
        kind = G._pick(rng, MIX)
        apply_fault(t, rng, kind)
        t.meta["fault_class"] = kind
        out.append(t)
    return out


if __name__ == "__main__":
    base = single("s4_lang", 150, 302)
    for lang in ("en", "hi", "hinglish", "para"):
        ts = []
        for t in base:
            u = copy.deepcopy(t)
            u.task_id, u.split = t.task_id.replace("s4_lang", f"s4_lang_{lang}"), f"s4_lang_{lang}"
            u.prompt = prompt_variant(t, lang)
            u.meta["lang"] = lang
            ts.append(u)
        T.write(f"s4_lang_{lang}", ts) if lang == "en" else None
        if lang != "en":  # same worlds as s4_lang_en by design: write without the PO-disjointness check against it
            from pathlib import Path
            import json
            (G.DATA / f"s4_lang_{lang}.jsonl").write_text("".join(json.dumps(x.to_json(), sort_keys=True) + "\n" for x in ts))
            print(f"s4_lang_{lang}", len(ts), "(paired with s4_lang_en)")
    T.write("s4_api", single("s4_api", 150, 301))
    T.write("s4_wf4", single("s4_wf4", 100, 303, make=workflow4_task))
    rng = random.Random(304)
    T.write("s4_combo", [combo_task("s4_combo", i, rng) for i in range(100)])
