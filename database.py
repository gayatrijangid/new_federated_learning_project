"""
database.py
============
Tiny SQLite store for simulation history, so every run is saved and can be
re-opened and replayed later from the UI ("History" tab).

Uses only Python's built-in sqlite3 -- nothing extra to install.
The database file (fl_runs.db) is created next to this file on first use.
"""
import json
import os
import sqlite3
from datetime import datetime, timezone

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fl_runs.db")


def _connect():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    with _connect() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS runs (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at  TEXT NOT NULL,
                config      TEXT NOT NULL,
                best_method TEXT,
                best_acc    REAL,
                events      TEXT NOT NULL
            )
        """)


def save_run(config: dict, events: list, final_metrics: dict) -> int:
    """Store one finished simulation (config + every streamed event)."""
    best_method, best_acc = None, None
    for method, m in (final_metrics or {}).items():
        acc = m["classification"]["accuracy"]
        if best_acc is None or acc > best_acc:
            best_method, best_acc = method, acc

    with _connect() as conn:
        cur = conn.execute(
            "INSERT INTO runs (created_at, config, best_method, best_acc, events) VALUES (?,?,?,?,?)",
            (datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
             json.dumps(config), best_method, best_acc, json.dumps(events)),
        )
        return cur.lastrowid


def list_runs(limit: int = 50) -> list:
    with _connect() as conn:
        rows = conn.execute(
            "SELECT id, created_at, config, best_method, best_acc FROM runs ORDER BY id DESC LIMIT ?",
            (limit,),
        ).fetchall()
    return [{
        "id": r["id"],
        "created_at": r["created_at"],
        "config": json.loads(r["config"]),
        "best_method": r["best_method"],
        "best_acc": r["best_acc"],
    } for r in rows]


def get_run(run_id: int):
    with _connect() as conn:
        r = conn.execute("SELECT * FROM runs WHERE id = ?", (run_id,)).fetchone()
    if r is None:
        return None
    return {
        "id": r["id"],
        "created_at": r["created_at"],
        "config": json.loads(r["config"]),
        "events": json.loads(r["events"]),
    }


def delete_run(run_id: int) -> bool:
    with _connect() as conn:
        cur = conn.execute("DELETE FROM runs WHERE id = ?", (run_id,))
        return cur.rowcount > 0
