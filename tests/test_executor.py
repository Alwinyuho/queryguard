import hashlib

import pytest

from queryguard.executor import execute, results_match

WRITES = [
    "DELETE FROM customer",
    "DROP TABLE customer",
    "UPDATE customer SET name = 'x'",
    "INSERT INTO customer VALUES (9, 'x', 'y')",
    "CREATE TABLE copy AS SELECT * FROM customer",
    "PRAGMA writable_schema = 1",
    "ATTACH DATABASE 'other.db' AS other",
]


def test_reads_work(shop_db):
    result = execute(shop_db, "SELECT name FROM customer ORDER BY id")
    assert result.ok
    assert result.rows == [("Asha",), ("Ravi",)]
    assert result.columns == ["name"]


@pytest.mark.parametrize("sql", WRITES)
def test_writes_fail_even_without_the_guard(shop_db, sql):
    before = hashlib.sha256(shop_db.read_bytes()).hexdigest()
    assert not execute(shop_db, sql).ok
    assert hashlib.sha256(shop_db.read_bytes()).hexdigest() == before


def test_runaway_query_is_stopped(shop_db):
    sql = "WITH RECURSIVE n(x) AS (SELECT 1 UNION ALL SELECT x + 1 FROM n) SELECT COUNT(*) FROM n"
    result = execute(shop_db, sql, timeout_s=0.5)
    assert not result.ok
    assert "time limit" in result.error


def test_row_cap(shop_db):
    result = execute(shop_db, "SELECT * FROM orders", max_rows=2)
    assert len(result.rows) == 2
    assert result.truncated


def test_results_match_ignores_row_order_like_bird():
    assert results_match([(1,), (2,)], [(2,), (1,)])
    assert not results_match([(1,)], [(1,), (2,)])
