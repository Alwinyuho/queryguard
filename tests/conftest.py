import sqlite3

import pytest

from queryguard import config


@pytest.fixture
def shop_db(tmp_path, monkeypatch):
    """A tiny SQLite database laid out like a BIRD database folder."""
    folder = tmp_path / "shop"
    (folder / "database_description").mkdir(parents=True)
    path = folder / "shop.sqlite"
    connection = sqlite3.connect(path)
    connection.executescript(
        """
        CREATE TABLE customer (id INTEGER PRIMARY KEY, name TEXT, city TEXT);
        CREATE TABLE orders (
            id INTEGER PRIMARY KEY,
            customer_id INTEGER REFERENCES customer(id),
            amount REAL
        );
        INSERT INTO customer VALUES (1, 'Asha', 'Chennai'), (2, 'Ravi', 'Pune');
        INSERT INTO orders VALUES (1, 1, 250.0), (2, 1, 100.0), (3, 2, 75.5);
        """
    )
    connection.commit()
    connection.close()
    (folder / "database_description" / "customer.csv").write_text(
        "original_column_name,column_name,column_description,data_format,value_description\n"
        "city,,city where the customer lives,text,\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(config, "DATABASES_DIR", tmp_path)
    return path
