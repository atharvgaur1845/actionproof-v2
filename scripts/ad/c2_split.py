"""AD-C2 sealed evaluation split cw_c2 (PREREGISTRATION_AD_C2.md): 300 tasks, seed 1204, same CW world prior as cw_conf.
Asserts instance-disjointness (task ids, PO ids, prompts) from cw_train, cw_dev and cw_conf, then writes and hashes.
    python scripts/ad/c2_split.py
"""
from __future__ import annotations

import hashlib, json
from pathlib import Path

from actionproof.ad.pools import make_pool
from actionproof.stage3.tasks import write

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data/bench/proc"


def main() -> None:
    out = DATA / "cw_c2.jsonl"
    if out.exists():
        raise SystemExit("cw_c2 exists; the split is written once")
    new = make_pool("cw_c2", 300, 1204)
    seen_ids, seen_po, seen_prompt = set(), set(), set()
    for name in ("cw_train", "cw_dev", "cw_conf"):
        for line in (DATA / f"{name}.jsonl").open():
            r = json.loads(line)
            seen_ids.add(r["task_id"]); seen_prompt.add(r["prompt"]); seen_po.add(r["meta"].get("po_id"))
    for t in new:
        assert t.task_id not in seen_ids and t.prompt not in seen_prompt and t.meta.get("po_id") not in seen_po, t.task_id
    write("cw_c2", new)
    h = hashlib.sha256(out.read_bytes()).hexdigest()
    classes = {}
    for t in new:
        classes[t.meta["fault_class"]] = classes.get(t.meta["fault_class"], 0) + 1
    (DATA / "cw_c2_manifest.json").write_text(json.dumps({"split": "cw_c2", "n": len(new), "seed": 1204,
                                                         "sha256": h, "fault_classes": classes}, indent=2))
    print(h, len(new), classes)


if __name__ == "__main__":
    main()
