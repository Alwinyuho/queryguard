"""Two independent safety layers for model-written SQL.

Layer 1 (this module, static): parse the SQL into a syntax tree and allow only a
single read-only SELECT. Checking the tree, not the text, means tricks such as
comments, odd casing or a second statement after a semicolon do not get through.

Layer 2 (executor.py, runtime): the database is opened read-only and SQLite's
authorizer denies every action except reading. Even if layer 1 had a bug, a write
would still fail.
"""

from __future__ import annotations

from dataclasses import dataclass

import sqlglot
from sqlglot import exp
from sqlglot.errors import SqlglotError

_FORBIDDEN_NAMES = (
    "Insert",
    "Update",
    "Delete",
    "Drop",
    "Create",
    "Alter",
    "Merge",
    "Command",
    "Pragma",
    "Transaction",
    "Commit",
    "Rollback",
    "Attach",
    "Detach",
    "Into",
)
# Look the classes up by name so the guard works across sqlglot versions.
FORBIDDEN_NODES = tuple(getattr(exp, n) for n in _FORBIDDEN_NAMES if hasattr(exp, n))
FORBIDDEN_FUNCTIONS = {"load_extension", "readfile", "writefile", "edit", "fts3_tokenizer"}
READ_ROOTS = (exp.Select, exp.Union, exp.Intersect, exp.Except)


@dataclass(frozen=True)
class GuardResult:
    allowed: bool
    reason: str = ""
    # "write": the query tries to change data or settings, so the agent refuses outright.
    # "malformed": it cannot be checked (parse error, several statements), so it can be repaired.
    kind: str = ""


def _write_reason(tree: exp.Expression) -> str:
    """Why this statement is not read-only, or an empty string if it is."""
    if not isinstance(tree, READ_ROOTS):
        return f"only SELECT queries are allowed, got {tree.key.upper()}"
    for node in tree.walk():
        if isinstance(node, FORBIDDEN_NODES):
            return f"forbidden operation: {node.key.upper()}"
        if isinstance(node, exp.Func):
            name = (node.name if isinstance(node, exp.Anonymous) else node.sql_name()).lower()
            if name in FORBIDDEN_FUNCTIONS:
                return f"forbidden function: {name}"
    return ""


def check_sql(sql: str) -> GuardResult:
    """Allow exactly one read-only query; explain the refusal otherwise."""
    if not sql or not sql.strip():
        return GuardResult(False, "empty query", "malformed")
    try:
        statements = [s for s in sqlglot.parse(sql, read="sqlite") if s is not None]
    except SqlglotError as error:  # covers both ParseError and TokenError
        first_line = str(error).splitlines()[0]
        return GuardResult(False, f"could not parse SQL: {first_line}", "malformed")
    for statement in statements:
        reason = _write_reason(statement)
        if reason:
            return GuardResult(False, reason, "write")
    if len(statements) != 1:
        return GuardResult(False, f"expected 1 statement, found {len(statements)}", "malformed")
    return GuardResult(True)
