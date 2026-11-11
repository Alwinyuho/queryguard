"""Sandboxed, read-only SQLite execution with a time limit and a row cap."""

from __future__ import annotations

import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path

from . import config

# SQLite authorizer action codes that a read-only query needs.
_ALLOWED_ACTIONS = {
    sqlite3.SQLITE_SELECT,
    sqlite3.SQLITE_READ,
    sqlite3.SQLITE_FUNCTION,
    getattr(sqlite3, "SQLITE_RECURSIVE", 33),
}


def _authorizer(action, arg1, arg2, db_name, trigger):
    return sqlite3.SQLITE_OK if action in _ALLOWED_ACTIONS else sqlite3.SQLITE_DENY


@dataclass
class ExecResult:
    rows: list[tuple] = field(default_factory=list)
    columns: list[str] = field(default_factory=list)
    error: str | None = None
    truncated: bool = False
    elapsed_ms: float = 0.0

    @property
    def ok(self) -> bool:
        return self.error is None


def connect_read_only(path: Path) -> sqlite3.Connection:
    uri = f"{path.resolve().as_uri()}?mode=ro"
    connection = sqlite3.connect(uri, uri=True, check_same_thread=False)
    connection.set_authorizer(_authorizer)
    return connection


def execute(
    path: Path,
    sql: str,
    timeout_s: float = config.QUERY_TIMEOUT_S,
    max_rows: int = config.MAX_ROWS,
) -> ExecResult:
    started = time.perf_counter()
    deadline = started + timeout_s
    connection = connect_read_only(path)
    # Called every N virtual-machine steps; returning non-zero aborts the query.
    connection.set_progress_handler(lambda: int(time.perf_counter() > deadline), 10_000)
    try:
        cursor = connection.execute(sql)
        rows = cursor.fetchmany(max_rows + 1)
        columns = [d[0] for d in cursor.description or []]
        truncated = len(rows) > max_rows
        return ExecResult(
            rows=[tuple(r) for r in rows[:max_rows]],
            columns=columns,
            truncated=truncated,
            elapsed_ms=(time.perf_counter() - started) * 1000,
        )
    except sqlite3.Error as error:
        message = str(error)
        if time.perf_counter() > deadline:
            message = f"query exceeded the {timeout_s:.0f}s time limit"
        return ExecResult(error=message, elapsed_ms=(time.perf_counter() - started) * 1000)
    finally:
        connection.close()


def results_match(predicted: list[tuple], gold: list[tuple]) -> bool:
    """BIRD's official rule: the two results contain the same set of rows."""
    return set(predicted) == set(gold)
