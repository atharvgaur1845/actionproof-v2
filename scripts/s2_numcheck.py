"""Check that every published number resolves through claims/claims.yaml.

1. Each `numbers` entry is re-read from its file (key path, optional mean/sum) and compared with its `value`, rounded
   to the decimals written in `value`.
2. Each document under `documents` is scanned for result-like tokens (a decimal point, a percent sign, thousands
   commas or an exponent). Every token must be one of the `shown` strings of a claim that is not pending, or be listed
   under `allow` (non-result numbers such as section times or version numbers).
Integers without commas are not scanned. Exit code 1 on any failure.
    python scripts/s2_numcheck.py
"""
from __future__ import annotations

import json
import re
import sys
from decimal import Decimal
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
TOKEN = re.compile(r"(?<![\w./:-])[-+]?(?:\d{1,3}(?:,\d{3})+|\d+\.\d+|\d+)(?:e-?\d+)?%?(?![\w/:])")


def resolve(obj: Any, path: list) -> Any:
    for p in path:
        if isinstance(p, dict):
            obj = next(x for x in obj if all(str(x.get(k)) == str(v) for k, v in p.items()))
        elif isinstance(obj, list):
            obj = obj[int(p)]
        else:
            obj = obj[p]
    return obj


def file_value(n: dict) -> float:
    if isinstance(n["file"], list):  # the same key in several files (e.g. one per training seed), averaged
        vals = [file_value({**n, "file": f, "agg": None}) for f in n["file"]]
        return sum(vals) / len(vals)
    text = (ROOT / n["file"]).read_text()
    key = n["key"]
    if n["file"].endswith(".md"):  # markdown result tables: `key` is a regex whose first group is the number
        m = re.search(key, text)
        if not m:
            raise KeyError(f"pattern not found: {key}")
        return float(m.group(1).replace(",", ""))
    data = json.loads(text)
    if n.get("agg") == "sum" and key and isinstance(key[0], list):
        return float(sum(resolve(data, k) for k in key))
    v = resolve(data, key)
    if n.get("agg") == "mean":
        return sum(v) / len(v)
    if n.get("agg") == "sum":
        return float(sum(v))
    return float(v)


def matches(stated: Any, actual: float) -> bool:
    s = str(stated)
    if "e" in s.lower():  # scientific: compare to 2 significant digits
        return float(f"{actual:.1e}") == float(f"{float(s):.1e}")
    places = -Decimal(s).as_tuple().exponent if "." in s else 0
    return round(actual, max(places, 0)) == round(float(s), max(places, 0))


def is_result_token(t: str) -> bool:
    return any(c in t for c in ".%,e") and not t.endswith(".")


def main() -> int:
    cfg = yaml.safe_load((ROOT / "claims/claims.yaml").read_text())
    fails, shown = [], set()
    for c in cfg["claims"]:
        for name, n in (c.get("numbers") or {}).items():
            try:
                actual = file_value(n)
            except Exception as e:  # noqa: BLE001 - report any unresolvable path
                fails.append(f"{c['id']}.{name}: cannot read {n['file']} {n['key']}: {e!r}")
                continue
            if not matches(n["value"], actual):
                fails.append(f"{c['id']}.{name}: stated {n['value']} but file gives {actual}")
            if c["status"] != "pending":
                shown.update(n.get("shown", []))
    allow = set(map(str, cfg.get("allow", [])))
    for doc in cfg["documents"]:
        text = (ROOT / doc).read_text()
        for i, line in enumerate(text.splitlines(), 1):
            line = re.sub(r"`[^`]*`", "", line)  # paths and commands are not claims
            for t in TOKEN.findall(line):
                if is_result_token(t) and t not in shown and t not in allow:
                    fails.append(f"{doc}:{i}: number {t!r} is not in claims.yaml")
    for f in fails:
        print("FAIL", f)
    n_num = sum(len(c.get("numbers") or {}) for c in cfg["claims"])
    print(f"{n_num} numbers checked against files, {len(cfg['documents'])} documents scanned, {len(fails)} failures")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
