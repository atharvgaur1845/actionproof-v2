"""Mutation testing of the TXN verifier and environment: seeded mutants of actionproof/txn/truth.py, env.py and
invariants.py, each applied in place, tested with `pytest -x tests/txn`, then reverted. A mutant is killed when the
suite fails. Every mutant must be killed; the run exits 1 otherwise.

The original files are restored after every mutant (and on interrupt). A sha256 check before and after confirms it.

    python scripts/verifier_mutation.py [--out results/txn/mutation_report.json] [--only M03]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import signal
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ENV, TRUTH, INV = (ROOT / "actionproof/txn" / f for f in ("env.py", "truth.py", "invariants.py"))

# (id, file, original text, mutated text, description); each original occurs exactly once in its file
MUTANTS = [
    ("M01", ENV, "LATE_DELAY = 2", "LATE_DELAY = 1", "late commit off by one (lands after the 1st later call)"),
    ("M02", ENV, "LATE_DELAY = 2", "LATE_DELAY = 3", "late commit off by one (lands after the 3rd later call)"),
    ("M03", ENV, 'if p["created_idx"] == idx:', "if False:", "the faulted call itself counts towards the landing"),
    ("M04", ENV, 'if p["scope"] is not None and p["scope"] in self.idem:',
     'if p["scope"] is not None and any(s[:2] == p["scope"][:2] for s in self.idem):',
     "late-commit dedupe ignores the idempotency key (tool, po only)"),
    ("M05", ENV, '            self.idem[p["scope"]] = {"status": "ok", p["id_field"]: p["new_id"]}',
     "            pass", "a landed late commit is not stored for later same-key retries"),
    ("M06", ENV, 'audit_status = "error" if kind == "log_conflict" else "timeout"', 'audit_status = "timeout"',
     "log_conflict: the audit log records the timeout instead of the conflicting error"),
    ("M07", ENV, "                if scope:\n                    self.idem[scope] = {\"status\": \"ok\", id_field: new_id}",
     "                pass", "log_conflict/visibility_lag commit without storing the key"),
    ("M08", ENV, "                    self._colleague(tool, args)", "                    pass",
     "concurrent: the colleague's write is never inserted"),
    ("M09", ENV, "        if consistent:\n            return self.state", "        if False:\n            return self.state",
     "the consistent flag is ignored"),
    ("M10", ENV, "LAG_READS = 2", "LAG_READS = 1", "delayed window off by one (1 stale read)"),
    ("M11", ENV, "        if AUDIT_TOOL in self.unavailable:", "        if False:",
     "the audit log stays available in the unresolvable regime"),
    ("M12", ENV, '            return self.state, self.version[service], False\n        w = self.lag.get(service)',
     '            return self.state, self.version[service], True\n        w = self.lag.get(service)',
     "a consistent read is labelled stale"),
    ("M13", TRUTH, 'if p["kind"] == "payment" and p["po_id"] == pid\n                     and p["payment_id"] not in refunded',
     'if p["kind"] in ("payment", "refund") and p["po_id"] == pid\n                     and p["payment_id"] not in refunded',
     "FCompR counts a refund row as a live payment"),
    ("M14", TRUTH, 'commit_idx(b) < a["idx"]', 'commit_idx(b) <= a["idx"]',
     "URR: a retry concurrent with a late landing counts as after it"),
    ("M15", TRUTH, '        if a["from_idempotency_store"]:\n            continue\n', "",
     "URR counts idempotency-store replies as re-executions"),
    ("M16", TRUTH, '"str": vtc or (esc and esc_ok and not harm)', '"str": vtc or (esc and esc_ok)',
     "STR ignores harm caused before escalating"),
    ("M17", TRUTH, '"str": vtc or (esc and esc_ok and not harm)', '"str": vtc or (esc and not harm)',
     "STR ignores whether escalation was correct"),
    ("M18", TRUTH, 'fcr = status == "completed" and not base["goal_met"]', 'fcr = not base["goal_met"]',
     "FCR counts honest failures"),
    ("M19", TRUTH, 'verdicts.get(ps) == VERIFIED and not dup', 'verdicts.get(ps) == VERIFIED',
     "PRS ignores duplicates"),
    ("M20", TRUTH, 'unres and status == "completed" and bool(base["goal_met"])', 'unres and status == "completed"',
     "unverified claim does not require goal_met"),
    ("M21", TRUTH, 'if (rec.get("report") or {}).get("status") == "escalated":',
     'if (rec.get("report") or {}).get("status") in ("escalated", "failed"):', "a failed report counts as escalation"),
    ("M22", TRUTH, 'fcompr = any(n == 1 and was for _, n, was in refunds)', 'fcompr = any(n >= 1 and was for _, n, was in refunds)',
     "FCompR flags refunds of an extra payment"),
    ("M23", TRUTH, '    dup = any(v == DUPLICATE for v in verdicts.values())\n    wrong',
     '    dup = verdicts.get("pay") == DUPLICATE\n    wrong', "DMR looks only at the pay step"),
    ("M24", INV, "            replay -= p[\"amount_cents\"]", "            replay += p[\"amount_cents\"]",
     "money conservation counts a refund as a payment"),
    ("M25", INV, 'if k2 is None or k2 != k1:', 'if k2 is not None and k2 != k1:',
     "unresolved-mutation check exempts unkeyed retries"),
]


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def run_suite(timeout: int = 600) -> tuple[int, float, str]:
    t0 = time.time()
    # a fresh bytecode cache per run: a same-size mutant written within the same second would otherwise be served from
    # the stale .pyc (pyc validation uses source mtime in seconds and size)
    with tempfile.TemporaryDirectory() as cache:
        env = {**os.environ, "PYTHONPYCACHEPREFIX": cache, "PYTHONDONTWRITEBYTECODE": "1"}
        p = subprocess.run([sys.executable, "-m", "pytest", "-q", "-x", "-p", "no:cacheprovider", "tests/txn"],
                           cwd=ROOT, capture_output=True, text=True, timeout=timeout, env=env)
    tail = (p.stdout.strip().splitlines() or [""])[-1]
    return p.returncode, time.time() - t0, tail


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "results/txn/mutation_report.json"))
    ap.add_argument("--only", default=None)
    a = ap.parse_args()
    files = {ENV, TRUTH, INV}
    originals = {f: f.read_text() for f in files}
    before = {f: sha(f) for f in files}

    def restore(*_):
        for f, text in originals.items():
            f.write_text(text)
    signal.signal(signal.SIGINT, lambda *x: (restore(), sys.exit(130)))
    signal.signal(signal.SIGTERM, lambda *x: (restore(), sys.exit(143)))
    for mid, f, old, new, _ in MUTANTS:
        n = originals[f].count(old)
        assert n == 1, f"{mid}: original text occurs {n} times in {f.name}"
    code, secs, tail = run_suite()
    if code != 0:
        sys.exit(f"the unmutated suite fails ({tail}); fix it before mutation testing")
    results = []
    try:
        for mid, f, old, new, desc in MUTANTS:
            if a.only and mid != a.only:
                continue
            f.write_text(originals[f].replace(old, new))
            try:
                code, secs, tail = run_suite()
            finally:
                restore()
            killed = code != 0
            results.append({"id": mid, "file": str(f.relative_to(ROOT)), "description": desc, "killed": killed,
                            "pytest": tail, "seconds": round(secs, 1)})
            print(f"{mid} {'KILLED ' if killed else 'SURVIVED'} {desc} [{tail}]", flush=True)
    finally:
        restore()
    assert {f: sha(f) for f in files} == before, "files not restored"
    k = sum(r["killed"] for r in results)
    rep = {"mutants": len(results), "killed": k, "kill_rate": k / len(results) if results else None,
           "suite": "pytest -x tests/txn", "results": results}
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rep, indent=1))
    print(json.dumps({k_: rep[k_] for k_ in ("mutants", "killed", "kill_rate")}))
    return 0 if k == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
