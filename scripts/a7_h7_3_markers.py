"""H7-3 (PREREGISTRATION_A7 §5): per episode on `e_heldout`, does the refund-key format predict credit omission?

Marker: the first refund's idempotency key equals `refund-<order>` (the A6 demonstrator's format).
Prediction: marker present -> the episode never calls `apply_store_credit`; marker absent -> it does.
Registered threshold: accuracy >= 0.9 across the A6 models (below 0.8 the two-template description is not used).
Episodes without a refund call count as marker absent. Also reported for `e_heldout_ctrl` (not registered).

    python scripts/a7_h7_3_markers.py results/eval/a6 > results/a7/h7_3_markers.md
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

MODELS = ("init_a6", "a6_pcstr_d_s0", "a6_pcstr_d_noreport_s0", "a6_pcstr_d_noreport_s1", "a6_pcstr_d_noND_s0",
          "a6_outcome_s0", "a6_ack_outcome_s0", "a6_pcstr_d_noFunder_s0", "a6_pcstr_d_s1")


def _args(s):
    return s.get("args") if isinstance(s.get("args"), dict) else {}


def row(r: dict) -> tuple[bool, bool]:
    t = [s for s in r["trace"] if not s.get("by")]
    ref = next((s for s in t if s["tool"] == "issue_refund"), None)
    marker = bool(ref) and _args(ref).get("idempotency_key") == f"refund-{_args(ref).get('order_id')}"
    omitted = not any(s["tool"] == "apply_store_credit" for s in t)
    return marker, omitted


def main() -> None:
    root = Path(sys.argv[1] if len(sys.argv) > 1 else "results/eval/a6")
    for split, registered in (("e_heldout", True), ("e_heldout_ctrl", False)):
        print(f"## `{split}`" + (" (registered H7-3)" if registered else " (not registered)") + "\n")
        print("| model | n | marker | credit omitted | marker predicts omission |")
        print("|---|---|---|---|---|")
        hit = tot = 0
        for m in MODELS:
            p = root / f"{m}__neutral__{split}.jsonl"
            if not p.exists():
                print(f"| {m} | – | – | – | trace not available |")
                continue
            rows = [row(json.loads(l)) for l in open(p)]
            h = sum(a == b for a, b in rows)
            hit, tot = hit + h, tot + len(rows)
            print(f"| {m} | {len(rows)} | {sum(a for a, _ in rows)} | {sum(b for _, b in rows)} | {h}/{len(rows)} |")
        print(f"\n**Accuracy: {hit}/{tot} = {hit / tot:.3f}**\n" if tot else "")


if __name__ == "__main__":
    main()
