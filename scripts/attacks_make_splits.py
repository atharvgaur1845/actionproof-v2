"""Write the attack-suite development split and the manifest (docs/attacks/SPEC.md §5).

    python scripts/attacks_make_splits.py            # data/attacks/attack_dev.jsonl + data/attacks/attack_manifest.json
    python scripts/attacks_make_splits.py --heldout-out /scratch/.../attack_heldout.jsonl   # optional file copy

The held-out split is not written into the repository: it is regenerated from its seed (generate.make_split) and
checked against the sha256 in the manifest. Nothing here runs a policy on any split.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main() -> None:
    from actionproof.attacks import spec as S
    from actionproof.attacks.generate import MANIFEST, OUT, check_disjoint, counts, make_split, split_hash, task_line
    ap = argparse.ArgumentParser()
    ap.add_argument("--heldout-out", default=None, help="also write the held-out tasks to this path (not committed)")
    ap.add_argument("--force", action="store_true", help="overwrite an existing manifest")
    a = ap.parse_args()
    if MANIFEST.exists() and not a.force:
        raise SystemExit(f"{MANIFEST} exists; the splits are registered (use --force only before registration)")
    splits = {s: make_split(s) for s in ("attack_dev", "attack_heldout")}
    disjoint = check_disjoint(splits)
    again = make_split("attack_heldout")
    assert split_hash(again) == split_hash(splits["attack_heldout"]), "held-out generation is not deterministic"
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "attack_dev.jsonl").write_text("".join(task_line(t) for t in splits["attack_dev"]))
    if a.heldout_out:
        Path(a.heldout_out).write_text("".join(task_line(t) for t in splits["attack_heldout"]))
    man = {"suite": S.SUITE, "spec": "docs/attacks/SPEC.md", "seeds": S.SEEDS, "sizes": S.SIZES,
           "generator": "actionproof/attacks/generate.py::make_split", "disjointness": disjoint,
           "templates": {"dev": {f: [S.template_id(f, c, p) for c, p in S.dev_templates(f)] for f in S.FAMILIES},
                         "heldout": {f: [S.template_id(f, c, p) for c, p in S.heldout_templates(f)]
                                     for f in S.FAMILIES},
                         "heldout_only_families": list(S.HELDOUT_ONLY_FAMILIES),
                         "compound_pairs": [list(p) for p in S.COMPOUND_PAIRS]},
           "splits": {}}
    for s, tasks in splits.items():
        man["splits"][s] = {"sha256": split_hash(tasks), "file": f"data/attacks/{s}.jsonl" if s == "attack_dev"
                            else None, **counts(tasks)}
    MANIFEST.write_text(json.dumps(man, indent=1, sort_keys=True, ensure_ascii=False) + "\n")
    print(json.dumps({s: {k: v for k, v in man["splits"][s].items() if k in ("n", "by_variant", "sha256")}
                      for s in splits}, indent=1))


if __name__ == "__main__":
    main()
