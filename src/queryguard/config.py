"""Paths and settings shared by the agent, the API and the evaluation scripts."""

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(os.environ.get("QUERYGUARD_HOME", Path(__file__).resolve().parents[2]))
BIRD_DIR = Path(os.environ.get("BIRD_DIR", ROOT.parent / "_downloads" / "bird" / "extracted"))
DATABASES_DIR = BIRD_DIR / "dev_databases"
REPORT_DIR = ROOT / "reports"
RUNS_DIR = ROOT / "runs"

QUESTIONS_URL = (
    "https://huggingface.co/datasets/birdsql/bird_mini_dev/resolve/main/data/mini_dev_sqlite.jsonl"
)

LOCAL_LLM_MODEL = os.environ.get("QUERYGUARD_LOCAL_MODEL", "Qwen/Qwen3-4B")
LOCAL_LLM_4BIT = os.environ.get("QUERYGUARD_LOCAL_4BIT", "1") == "1"
# Token budget for one generation batch (prompt + output, summed over the batch).
LOCAL_LLM_BATCH_TOKENS = int(os.environ.get("QUERYGUARD_BATCH_TOKENS", "12000"))
EMBED_MODEL = "BAAI/bge-small-en-v1.5"

MAX_REPAIRS = 2
QUERY_TIMEOUT_S = 30.0
MAX_ROWS = 10_000
SAMPLE_VALUES = 3
SCHEMA_TOP_COLUMNS = 30


def database_path(db_id: str) -> Path:
    """Return the SQLite file for a database id, rejecting path tricks."""
    if not db_id.replace("_", "").isalnum():
        raise ValueError(f"invalid database id: {db_id!r}")
    path = DATABASES_DIR / db_id / f"{db_id}.sqlite"
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def available_databases() -> list[str]:
    if not DATABASES_DIR.exists():
        return []
    return sorted(p.name for p in DATABASES_DIR.iterdir() if (p / f"{p.name}.sqlite").exists())
