import numpy as np

from queryguard.agent import QueryAgent, extract_sql
from queryguard.linking import select_columns
from queryguard.schema import load_schema


class ScriptedLLM:
    """Returns pre-written replies in order, recording what it was asked."""

    name = "scripted"

    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = []

    def generate_batch(self, batch):
        self.calls.append(batch[0])
        return [self.replies.pop(0)]


def test_extract_sql_from_code_block_and_plain_text():
    assert extract_sql("Here:\n```sql\nSELECT 1;\n```") == "SELECT 1"
    assert extract_sql("<think>hmm</think>SELECT 2") == "SELECT 2"


def test_schema_includes_keys_descriptions_and_examples(shop_db):
    text = load_schema("shop").render()
    assert "CREATE TABLE customer" in text
    assert "city where the customer lives" in text
    assert "FOREIGN KEY (customer_id) REFERENCES customer(id)" in text
    assert "Chennai" in text


def test_agent_repairs_after_an_error(shop_db):
    llm = ScriptedLLM(
        ["```sql\nSELECT nme FROM customer\n```", "```sql\nSELECT name FROM customer\n```"]
    )
    result = QueryAgent(llm).ask("List customer names", "shop")
    assert [s.status for s in result.steps] == ["error", "ok"]
    assert result.rows == [("Asha",), ("Ravi",)]
    assert "no such column" in llm.calls[1][-1]["content"]


def test_write_attempt_is_refused_without_a_repair_round(shop_db):
    llm = ScriptedLLM(["DROP TABLE customer", "SELECT name FROM customer"])
    result = QueryAgent(llm).ask("Drop the table", "shop")
    assert result.status == "refused_write"
    assert len(result.steps) == 1  # no quiet substitution of a different query
    assert result.rows == []


def test_malformed_sql_is_repaired(shop_db):
    llm = ScriptedLLM(["SELECT name FROM customer WHERE (", "SELECT name FROM customer"])
    result = QueryAgent(llm).ask("Names?", "shop")
    assert [s.status for s in result.steps] == ["guard_blocked", "ok"]


def test_linking_keeps_join_keys(shop_db):
    schema = load_schema("shop")
    similarity = np.array(
        [1.0 if c.name == "amount" or c.name == "name" else 0.0 for c in schema.columns]
    )
    keep = select_columns(schema, similarity, top_columns=2)
    assert ("orders", "customer_id") in keep
    assert ("customer", "id") in keep
