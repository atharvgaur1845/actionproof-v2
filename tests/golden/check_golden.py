"""Structural check for proc_golden.jsonl (stdlib only; no verdict logic).

Usage: python tests/golden/check_golden.py [path]
"""
import json
import os
import sys

STEPS = {"order", "pay", "receive", "post"}
CLASSES = {"VERIFIED_SUCCESS", "ABSENT", "PARTIAL", "DUPLICATE", "WRONG"}
EXPECTED_KEYS = {"steps", "collateral", "duplicate_irreversible", "goal_met", "vsc", "false_success", "r_v", "r_a",
                 "phantom_readback", "benign_readback", "lost_ack_readback", "precommit_recovery"}
TASK_KEYS = {"task_id", "split", "workflow", "prompt", "initial_state", "meta"}
META_KEYS = {"domain", "env_version", "po_id", "requested", "faults", "never_commit", "fault_class", "env_seed"}
TRACE_KEYS = {"idx", "tool", "args", "obs", "by"}
ATTEMPT_KEYS = {"idx", "tool", "po_id", "fault", "committed", "from_idempotency_store", "obs_status"}
TABLES = {"pos", "suppliers", "supplier_orders", "payments", "receipts", "journal", "escalations"}


def check(path):
    seen = set()
    n = 0
    with open(path) as f:
        for ln, line in enumerate(f, 1):
            c = json.loads(line)
            where = f"line {ln} ({c.get('case_id')})"
            assert {"case_id", "description", "episode", "expected"} <= set(c), where
            assert c["case_id"] not in seen, f"{where}: duplicate id"
            seen.add(c["case_id"])
            ep = c["episode"]
            assert set(ep) == {"task", "trace", "attempts", "report", "final_state"}, where
            task = ep["task"]
            assert set(task) == TASK_KEYS, where
            assert set(task["meta"]) == META_KEYS, where
            assert set(task["initial_state"]) == TABLES and set(ep["final_state"]) == TABLES, where
            req = task["meta"]["requested"]
            assert set(req) <= STEPS and req, where
            for i, t in enumerate(ep["trace"]):
                assert set(t) == TRACE_KEYS and t["idx"] == i and t["by"] in ("agent", "harness"), where
            idxs = {t["idx"]: t for t in ep["trace"]}
            for a in ep["attempts"]:
                assert set(a) == ATTEMPT_KEYS, where
                assert a["committed"] in ("none", "partial", "full", "double"), where
                assert a["obs_status"] == idxs[a["idx"]]["obs"]["status"], where
                assert idxs[a["idx"]]["tool"] == a["tool"], where
            rep = ep["report"]
            assert rep is None or rep["status"] in ("completed", "failed", "escalated"), where
            e = c["expected"]
            assert set(e) == EXPECTED_KEYS, where
            assert set(e["steps"]) == set(req) and set(e["steps"].values()) <= CLASSES, where
            for k in ("collateral", "duplicate_irreversible", "goal_met", "false_success"):
                assert isinstance(e[k], bool), where
            for k in ("vsc", "r_v", "r_a"):
                assert e[k] in (0, 1), where
            for k in ("phantom_readback", "benign_readback", "lost_ack_readback", "precommit_recovery"):
                num, den = e[k]
                assert isinstance(num, int) and isinstance(den, int) and 0 <= num <= den, where
            if c.get("ambiguous"):
                assert c.get("ambiguity_note"), where
            n += 1
    return n


if __name__ == "__main__":
    p = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(__file__), "proc_golden.jsonl")
    print(f"{check(p)} cases OK")
