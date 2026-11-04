"""SQLite world state for the e-commerce ops domain (Domain A).

Each rollout gets its own in-memory database built from a task's initial state, so
concurrent GRPO rollouts can never observe each other's side effects.
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
from typing import Any

SCHEMA = """
CREATE TABLE customers (customer_id TEXT PRIMARY KEY, name TEXT, tier TEXT);
CREATE TABLE orders (
    order_id TEXT PRIMARY KEY, customer_id TEXT, amount_cents INTEGER, status TEXT,
    payment_method TEXT, sku TEXT, qty INTEGER, delivered_days_ago INTEGER
);
CREATE TABLE refunds (
    refund_id TEXT PRIMARY KEY, order_id TEXT, amount_cents INTEGER, method TEXT,
    status TEXT, idempotency_key TEXT, seq INTEGER
);
CREATE TABLE tickets (ticket_id TEXT PRIMARY KEY, customer_id TEXT, order_id TEXT, status TEXT, note TEXT);
CREATE TABLE audit_notes (note_id TEXT PRIMARY KEY, ticket_id TEXT, text TEXT, seq INTEGER);
CREATE TABLE inventory (sku TEXT PRIMARY KEY, on_hand INTEGER);
CREATE TABLE escalations (escalation_id TEXT PRIMARY KEY, ticket_id TEXT, reason TEXT, seq INTEGER);
"""

TABLES = ("customers", "orders", "refunds", "tickets", "audit_notes", "inventory", "escalations")

# Delta = one parameterised SQL statement. Receipts store the deltas they applied so the
# counterfactual replay can rebuild any subset of committed writes exactly.
Delta = tuple[str, tuple[Any, ...]]


def build_db(initial_state: dict[str, list[dict[str, Any]]]) -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:", isolation_level=None)  # autocommit; we manage txns
    conn.row_factory = sqlite3.Row
    conn.executescript(SCHEMA)
    for table, rows in initial_state.items():
        for row in rows:
            cols = ", ".join(row)
            marks = ", ".join("?" for _ in row)
            conn.execute(f"INSERT INTO {table} ({cols}) VALUES ({marks})", tuple(row.values()))
    return conn


def apply_deltas(conn: sqlite3.Connection, deltas: list[Delta]) -> None:
    conn.execute("BEGIN")
    try:
        for sql, params in deltas:
            conn.execute(sql, params)
        conn.execute("COMMIT")
    except Exception:
        conn.execute("ROLLBACK")
        raise


def snapshot(conn: sqlite3.Connection) -> dict[str, list[dict[str, Any]]]:
    """Canonical, order-independent copy of every table (used by the verifier and reward fns)."""
    out: dict[str, list[dict[str, Any]]] = {}
    extra = sorted(r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")
                   if r[0] not in TABLES)  # other domains' tables (Domain B), after Domain A's
    for table in (*TABLES, *extra):
        rows = [dict(r) for r in conn.execute(f"SELECT * FROM {table}")]
        out[table] = sorted(rows, key=lambda r: json.dumps(r, sort_keys=True))
    return out


def entity_hash(snap: dict[str, list[dict[str, Any]]], key: str, value: str) -> str:
    """Hash of every row, in any table, whose `key` column equals `value` (e.g. order_id)."""
    rows = [
        (table, row) for table in snap for row in snap[table] if row.get(key) == value
    ]
    return hashlib.sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest()[:16]
