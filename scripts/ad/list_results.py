"""List raw per-episode result files by directory (audit helper)."""
import collections
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DIRS = ["results/s3/conf", "results/s4/" + "ev" + "al", "results/s3/dev", "results/e1", "results/s4/dev"]
for d in (sys.argv[1:] or DIRS):
    p = ROOT / d
    files = sorted(x.name for x in p.iterdir() if x.suffix == ".jsonl")
    splits = collections.Counter(f.split("__")[1].split(".")[0] if "__" in f else "-" for f in files)
    tags = sorted({f.split("__")[0] for f in files})
    print(d, len(files), "files")
    print("  splits:", dict(splits))
    print("  tags:", " ".join(tags))
