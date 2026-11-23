"""Compare two evaluation files task by task: identical call sequences, identical outcomes (seeding diagnostic).

    python scripts/ad/compare_traces.py results/ad/dev/ad_awrb_s0__cw_dev.jsonl results/ad/dev/ad_awrb_s1__cw_dev.jsonl
"""
import json
import sys


def load(p):
    out = {}
    for l in open(p):
        r = json.loads(l)
        out[r["task_id"]] = r
    return out


a, b = load(sys.argv[1]), load(sys.argv[2])
same_calls = same_vsc = 0
texts_same = 0
for t in a:
    ca = [(s["tool"], json.dumps(s["args"], sort_keys=True)) for s in a[t]["trace"]]
    cb = [(s["tool"], json.dumps(s["args"], sort_keys=True)) for s in b[t]["trace"]]
    same_calls += ca == cb
    same_vsc += a[t]["vsc"] == b[t]["vsc"]
    ta = [m["content"] for m in a[t]["conversation"] if m["role"] == "assistant"]
    tb = [m["content"] for m in b[t]["conversation"] if m["role"] == "assistant"]
    texts_same += ta == tb
print(f"tasks {len(a)}; identical call sequences {same_calls}; identical assistant text {texts_same}; same VSC {same_vsc}")
