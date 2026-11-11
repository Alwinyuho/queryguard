import pytest

from queryguard.guard import check_sql

SAFE = [
    "SELECT name FROM customer",
    "select count(*) from orders where amount > 10",
    "WITH big AS (SELECT * FROM orders WHERE amount > 100) SELECT COUNT(*) FROM big",
    "SELECT name FROM customer UNION SELECT city FROM customer",
    "SELECT `Free Meal Count` FROM frpm",
    "SELECT name FROM customer -- delete comment is harmless",
    "SELECT 'DROP TABLE x' AS text_that_looks_scary",
]

UNSAFE = [
    "DELETE FROM customer",
    "DROP TABLE customer",
    "UPDATE customer SET name = 'x'",
    "INSERT INTO customer VALUES (3, 'x', 'y')",
    "SELECT 1; DROP TABLE customer",
    "ATTACH DATABASE 'x.db' AS x",
    "PRAGMA writable_schema = 1",
    "CREATE TABLE copy AS SELECT * FROM customer",
    "ALTER TABLE customer ADD COLUMN x TEXT",
    "WITH x AS (SELECT 1) DELETE FROM customer",
    "REPLACE INTO customer VALUES (1, 'x', 'y')",
    "VACUUM",
    "SELECT load_extension('evil')",
    "BEGIN; DELETE FROM customer; COMMIT",
    "DeLeTe /* sneaky */ FROM customer",
    "SELECT * INTO backup FROM customer",
    "",
    "   ",
    "SELEC name FRM customer WHERE",
    # Real model output that once crashed the guard: an unclosed backtick is a
    # tokenizer error, not a parse error, so it must be caught too.
    "SELECT name FROM customer WHERE `U-PRO = '33' OR `U-PRO` = '34'",
]


@pytest.mark.parametrize("sql", SAFE)
def test_safe_queries_pass(sql):
    assert check_sql(sql).allowed, check_sql(sql).reason


@pytest.mark.parametrize("sql", UNSAFE)
def test_unsafe_queries_are_blocked(sql):
    result = check_sql(sql)
    assert not result.allowed
    assert result.reason


@pytest.mark.parametrize(
    ("sql", "kind"),
    [
        ("DELETE FROM customer", "write"),
        ("SELECT 1; DROP TABLE customer", "write"),
        ("SELECT name FROM customer WHERE (", "malformed"),
        ("SELECT 1; SELECT 2", "malformed"),
    ],
)
def test_block_kind_separates_writes_from_broken_sql(sql, kind):
    assert check_sql(sql).kind == kind
