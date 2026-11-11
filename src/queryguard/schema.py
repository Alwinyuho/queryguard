"""Describe a database for the model: tables, columns, keys, descriptions, example values."""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

from . import config
from .executor import connect_read_only


@dataclass
class Column:
    table: str
    name: str
    type: str
    primary_key: bool = False
    description: str = ""
    values_note: str = ""
    examples: list[str] = field(default_factory=list)

    def document(self) -> str:
        """Text used for schema retrieval."""
        return " ".join(
            part for part in (self.table, self.name, self.description, self.values_note) if part
        )


@dataclass
class Schema:
    db_id: str
    columns: list[Column]
    foreign_keys: list[tuple[str, str, str, str]]  # (table, column, ref_table, ref_column)

    @property
    def tables(self) -> list[str]:
        return list(dict.fromkeys(c.table for c in self.columns))

    def render(self, keep: set[tuple[str, str]] | None = None) -> str:
        """Render CREATE TABLE statements with comments. ``keep`` limits the columns shown."""
        blocks = []
        for table in self.tables:
            columns = [
                c
                for c in self.columns
                if c.table == table and (keep is None or (c.table, c.name) in keep)
            ]
            if not columns:
                continue
            lines = []
            for c in columns:
                note = "; ".join(
                    p
                    for p in (
                        c.description,
                        c.values_note,
                        f"examples: {', '.join(c.examples)}" if c.examples else "",
                    )
                    if p
                )
                key = " PRIMARY KEY" if c.primary_key else ""
                lines.append(f"  {quote(c.name)} {c.type}{key},{f'  -- {note}' if note else ''}")
            fks = [
                f"  FOREIGN KEY ({quote(col)}) REFERENCES {quote(ref_t)}({quote(ref_c)})"
                for t, col, ref_t, ref_c in self.foreign_keys
                if t == table
            ]
            body = "\n".join(lines + fks).rstrip(",")
            blocks.append(f"CREATE TABLE {quote(table)} (\n{body}\n);")
        return "\n\n".join(blocks)


def quote(name: str) -> str:
    return f"`{name}`" if not name.replace("_", "").isalnum() else name


def _clean(text: str | None) -> str:
    return " ".join((text or "").split())[:200]


def _read_descriptions(directory: Path) -> dict[tuple[str, str], tuple[str, str]]:
    notes: dict[tuple[str, str], tuple[str, str]] = {}
    if not directory.exists():
        return notes
    for path in directory.glob("*.csv"):
        table = path.stem.lower()
        for encoding in ("utf-8-sig", "cp1252", "latin-1"):
            try:
                with path.open(encoding=encoding, newline="") as handle:
                    rows = list(csv.DictReader(handle))
                break
            except UnicodeDecodeError:
                continue
        for row in rows:
            name = _clean(row.get("original_column_name")).lower()
            if name:
                description = _clean(row.get("column_description") or row.get("column_name"))
                notes[(table, name)] = (description, _clean(row.get("value_description")))
    return notes


@lru_cache(maxsize=32)
def load_schema(db_id: str) -> Schema:
    path = config.database_path(db_id)
    notes = _read_descriptions(path.parent / "database_description")
    connection = connect_read_only(path)
    connection.set_authorizer(None)  # metadata pragmas are trusted code, not model output
    try:
        tables = [
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
            )
        ]
        columns, foreign_keys = [], []
        for table in tables:
            for _, name, col_type, _, _, pk in connection.execute(
                f"PRAGMA table_info({quote_sql(table)})"
            ):
                description, values_note = notes.get((table.lower(), name.lower()), ("", ""))
                if description.lower() == name.lower():
                    description = ""
                examples = [
                    str(v)[:40]
                    for (v,) in connection.execute(
                        f"SELECT DISTINCT {quote_sql(name)} FROM (SELECT {quote_sql(name)} "
                        f"FROM {quote_sql(table)} WHERE {quote_sql(name)} IS NOT NULL LIMIT 200) "
                        f"LIMIT {config.SAMPLE_VALUES}"
                    )
                ]
                columns.append(
                    Column(
                        table,
                        name,
                        col_type or "TEXT",
                        bool(pk),
                        description,
                        values_note,
                        examples,
                    )
                )
            for row in connection.execute(f"PRAGMA foreign_key_list({quote_sql(table)})"):
                foreign_keys.append((table, row[3], row[2], row[4] or row[3]))
    finally:
        connection.close()
    return Schema(db_id, columns, foreign_keys)


def quote_sql(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'
