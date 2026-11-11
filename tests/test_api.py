from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from queryguard.api import create_app


class FakeAgent:
    llm = SimpleNamespace(name="fake")

    def ask(self, question, db_id, evidence):
        return SimpleNamespace(to_dict=lambda: {"sql": "SELECT 1", "question": question})


@pytest.fixture
def client(shop_db):
    with TestClient(create_app(FakeAgent())) as test_client:
        yield test_client


def test_lists_databases(client):
    assert client.get("/databases").json() == ["shop"]


def test_ask(client):
    response = client.post("/ask", json={"question": " total? ", "db_id": "shop"})
    assert response.status_code == 200
    assert response.json()["question"] == "total?"


def test_unknown_database(client):
    assert client.post("/ask", json={"question": "x", "db_id": "../etc"}).status_code == 404


def test_blank_question(client):
    assert client.post("/ask", json={"question": "  ", "db_id": "shop"}).status_code == 422
