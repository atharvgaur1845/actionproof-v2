"""Compact extract of a txn_stats output: per-model all-episode rates, H4 STR per OOD group, descriptive pairs.

    python scripts/txn_compact.py results/txn/stats_all_descriptive__txn_test.json results/txn/compact_descriptive__txn_test.json
"""
import json
import sys
from pathlib import Path

src, dst = Path(sys.argv[1]), Path(sys.argv[2])
d = json.loads(src.read_text())
keep = ("VTC", "STR", "FCR", "DMR_all", "URR_all", "CER", "unsafe_unresolvable", "harm")
out = {"source": str(src), "models": {t: {k: m["all"][k] for k in keep} for t, m in d["models"].items()},
       "H4_STR": {t: {g: {"ood": r["STR"]["ood"]["rate"], "in_distribution": r["STR"]["in_distribution"]["rate"]}
                      for g, r in rows.items()} for t, rows in d["H4"].items()},
       "pairs_descriptive": d["pairs_descriptive"], "H3_supported": d["H3"].get("supported")}
dst.write_text(json.dumps(out, indent=1))
print(dst, dst.stat().st_size)
