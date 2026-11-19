"""Append-only evidence ledger with a SHA-256 hash chain.

Each event stores `prev_hash` and `hash = sha256(prev_hash || canonical_json(event without hash))`; the genesis
`prev_hash` is 64 zeros. `verify()` recomputes the chain and returns the first bad sequence number, so any
modification, deletion, insertion or reordering of stored events is detected *relative to a trusted head*.
`anchor()` appends (seq, head) to a separate anchor log; with anchors kept outside this store, truncation or a full
rewrite after the last anchor is detected too. This is tamper detection under those assumptions, not tamper-proof
storage, and no signatures are made. SQLite triggers reject UPDATE and DELETE as defence in depth only.
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

GENESIS = "0" * 64
ACTORS = ("agent", "gateway", "verifier", "recovery", "human", "external")


def canonical(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)


def event_hash(prev_hash: str, body: dict[str, Any]) -> str:
    return hashlib.sha256((prev_hash + canonical(body)).encode()).hexdigest()


@dataclass(frozen=True)
class Event:
    seq: int
    tx_id: str
    op_id: str | None
    kind: str
    actor: str
    t_logical: int
    t_wall: float
    body: dict[str, Any]
    prev_hash: str
    hash: str

    def unhashed(self) -> dict[str, Any]:
        return {"seq": self.seq, "tx_id": self.tx_id, "op_id": self.op_id, "kind": self.kind, "actor": self.actor,
                "t_logical": self.t_logical, "t_wall": self.t_wall, "body": self.body, "prev_hash": self.prev_hash}

    def to_json(self) -> dict[str, Any]:
        return {**self.unhashed(), "hash": self.hash}


class EvidenceLedger:
    """In memory by default; pass `path` for a SQLite file. Subscribers receive every appended event (for SSE)."""

    def __init__(self, path: str | Path | None = None) -> None:
        self.conn = sqlite3.connect(str(path) if path else ":memory:", check_same_thread=False)
        self.conn.executescript("""
            CREATE TABLE IF NOT EXISTS events (seq INTEGER PRIMARY KEY, tx_id TEXT, op_id TEXT, kind TEXT, actor TEXT,
              t_logical INTEGER, t_wall REAL, body TEXT, prev_hash TEXT, hash TEXT);
            CREATE TABLE IF NOT EXISTS anchors (seq INTEGER, head TEXT, t_wall REAL);
            CREATE TRIGGER IF NOT EXISTS no_update BEFORE UPDATE ON events BEGIN SELECT RAISE(ABORT, 'append-only'); END;
            CREATE TRIGGER IF NOT EXISTS no_delete BEFORE DELETE ON events BEGIN SELECT RAISE(ABORT, 'append-only'); END;
        """)
        self.subscribers: list[Any] = []
        self._clock = 0

    def head(self) -> tuple[int, str]:
        r = self.conn.execute("SELECT seq, hash FROM events ORDER BY seq DESC LIMIT 1").fetchone()
        return (r[0], r[1]) if r else (0, GENESIS)

    def append(self, tx_id: str, kind: str, actor: str, body: dict[str, Any], op_id: str | None = None) -> Event:
        if actor not in ACTORS:
            raise ValueError(f"unknown actor {actor}")
        seq, prev = self.head()
        self._clock += 1
        ev = {"seq": seq + 1, "tx_id": tx_id, "op_id": op_id, "kind": kind, "actor": actor, "t_logical": self._clock,
              "t_wall": round(time.time(), 3), "body": json.loads(canonical(body)), "prev_hash": prev}
        e = Event(**ev, hash=event_hash(prev, ev))
        self.conn.execute("INSERT INTO events VALUES (?,?,?,?,?,?,?,?,?,?)",
                          (e.seq, e.tx_id, e.op_id, e.kind, e.actor, e.t_logical, e.t_wall, canonical(e.body),
                           e.prev_hash, e.hash))
        self.conn.commit()
        for s in list(self.subscribers):
            s(e)
        return e

    def events(self, tx_id: str | None = None, after: int = 0) -> Iterator[Event]:
        q = "SELECT * FROM events WHERE seq > ?" + (" AND tx_id = ?" if tx_id else "") + " ORDER BY seq"
        for r in self.conn.execute(q, (after, tx_id) if tx_id else (after,)):
            yield Event(r[0], r[1], r[2], r[3], r[4], r[5], r[6], json.loads(r[7]), r[8], r[9])

    def verify(self) -> dict[str, Any]:
        """Recompute the chain. `first_bad_seq` is the first event whose link or content does not match."""
        prev, n, expect = GENESIS, 0, 1
        for e in self.events():
            n += 1
            if e.seq != expect or e.prev_hash != prev or event_hash(prev, e.unhashed()) != e.hash:
                return {"ok": False, "n": n, "first_bad_seq": e.seq, "head": prev}
            prev, expect = e.hash, expect + 1
        anchors = self.conn.execute("SELECT seq, head FROM anchors ORDER BY seq").fetchall()
        hashes = {e.seq: e.hash for e in self.events()}
        for seq, head in anchors:
            if hashes.get(seq) != head:
                return {"ok": False, "n": n, "first_bad_seq": seq, "head": prev, "anchor_mismatch": True}
        return {"ok": True, "n": n, "first_bad_seq": None, "head": prev}

    def anchor(self, path: str | Path | None = None) -> dict[str, Any]:
        """Record the current head; with `path`, also append it to an external anchor file."""
        seq, head = self.head()
        self.conn.execute("INSERT INTO anchors VALUES (?,?,?)", (seq, head, time.time()))
        self.conn.commit()
        if path:
            with open(path, "a") as f:
                f.write(canonical({"seq": seq, "head": head, "t": time.time()}) + "\n")
        return {"seq": seq, "head": head}

    def tamper(self, seq: int, body: dict[str, Any]) -> None:
        """Test mode only: overwrite one stored event body behind the triggers (for demonstrations and tests)."""
        self.conn.execute("DROP TRIGGER IF EXISTS no_update")
        self.conn.execute("UPDATE events SET body = ? WHERE seq = ?", (canonical(body), seq))
        self.conn.execute("CREATE TRIGGER no_update BEFORE UPDATE ON events BEGIN SELECT RAISE(ABORT, 'append-only'); END;")
        self.conn.commit()
