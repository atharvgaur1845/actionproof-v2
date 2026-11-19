"""A11 H4 labelling tool (blind). Shows each item once; you answer whether every change the customer asked for
actually happened. Never shows the truth or any judge score. Resumes where you stopped.

    python scripts/h4_label.py --annotator A1
Answers: y = yes, everything requested really happened; n = no; u = cannot tell. Then a confidence from 1 to 3.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--annotator", required=True)
    a = ap.parse_args()
    items = [json.loads(l) for l in open(ROOT / "data/h4/items.jsonl")]
    out = ROOT / f"results/h4/labels_{a.annotator}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    done = {json.loads(l)["item"] for l in open(out)} if out.exists() else set()
    for it in items:
        if it["item"] in done:
            continue
        print("\n" + "=" * 100 + f"\nITEM {it['item'] + 1} of {len(items)}\n" + "=" * 100 + "\n" + it["text"])
        ans = ""
        while ans not in ("y", "n", "u"):
            ans = input("\nDid every change the customer asked for actually happen? [y/n/u]: ").strip().lower()
        conf = ""
        while conf not in ("1", "2", "3"):
            conf = input("Confidence 1 (guess) / 2 / 3 (sure): ").strip()
        with out.open("a") as f:
            f.write(json.dumps({"item": it["item"], "answer": ans, "confidence": int(conf), "annotator": a.annotator,
                                "time": time.strftime("%Y-%m-%dT%H:%M:%S")}) + "\n")
    print(f"\nAll {len(items)} items labelled. Saved to {out}")


if __name__ == "__main__":
    main()
