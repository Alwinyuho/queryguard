"""The agent loop: write SQL, check it, run it, and repair it from the error message.

The loop is split into small pure steps (``initial_messages``, ``check``,
``repair_messages``) so the online agent and the batched evaluation share
exactly the same logic.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field

from . import config
from .executor import ExecResult, execute
from .guard import check_sql
from .linking import SchemaLinker
from .llm import LLM, Messages
from .schema import load_schema

SYSTEM_PROMPT = (
    "You are an expert data analyst who writes SQLite queries. "
    "Write exactly one read-only SQLite SELECT query that answers the question. "
    "Use only tables and columns from the schema. Quote names containing spaces with backticks. "
    "Return only the columns the question asks for. "
    "Reply with the SQL inside a ```sql code block and nothing else."
)
_SQL_BLOCK = re.compile(r"```(?:sql|sqlite)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)
_THINK = re.compile(r"<think>.*?</think>", re.DOTALL)


@dataclass
class Step:
    attempt: int
    sql: str
    status: str  # ok | empty | error | guard_blocked (malformed) | refused_write
    detail: str = ""
    rows: int = 0


@dataclass
class AgentResult:
    question: str
    db_id: str
    sql: str
    status: str
    columns: list[str] = field(default_factory=list)
    rows: list[tuple] = field(default_factory=list)
    steps: list[Step] = field(default_factory=list)

    def to_dict(self, max_rows: int = 50) -> dict:
        data = asdict(self)
        data["rows"] = [list(r) for r in self.rows[:max_rows]]
        data["total_rows"] = len(self.rows)
        return data


def extract_sql(reply: str) -> str:
    reply = _THINK.sub("", reply).strip()
    match = _SQL_BLOCK.search(reply)
    sql = match.group(1) if match else reply
    return sql.strip().rstrip(";").strip()


def initial_messages(question: str, evidence: str, schema_text: str) -> Messages:
    hint = f"\nHint: {evidence}" if evidence else ""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Schema:\n{schema_text}\n\nQuestion: {question}{hint}"},
    ]


def check(db_id: str, sql: str, attempt: int) -> tuple[Step, ExecResult | None]:
    """Run the guard, then the sandboxed query, and classify the outcome."""
    verdict = check_sql(sql)
    if not verdict.allowed:
        # A write attempt is refused outright. Asking the model to "repair" it would only
        # make it answer a different, read-only question without telling the user.
        status = "refused_write" if verdict.kind == "write" else "guard_blocked"
        return Step(attempt, sql, status, verdict.reason), None
    result = execute(config.database_path(db_id), sql)
    if not result.ok:
        return Step(attempt, sql, "error", result.error or ""), result
    if not result.rows:
        return Step(attempt, sql, "empty", "query returned no rows"), result
    return Step(attempt, sql, "ok", rows=len(result.rows)), result


def needs_repair(step: Step) -> bool:
    return step.status in {"error", "guard_blocked", "empty"}


def repair_messages(messages: Messages, reply: str, step: Step) -> Messages:
    if step.status == "empty":
        feedback = (
            "The query ran but returned no rows. Check filter values against the example "
            "values in the schema, then write a corrected query."
        )
    elif step.status == "guard_blocked":
        feedback = f"The query was rejected: {step.detail}. Write a single read-only SELECT."
    else:
        feedback = f"The query failed with this error: {step.detail}. Write a corrected query."
    return messages + [
        {"role": "assistant", "content": reply},
        {"role": "user", "content": feedback},
    ]


class QueryAgent:
    def __init__(self, llm: LLM, use_linking: bool = False, max_repairs: int = config.MAX_REPAIRS):
        self.llm = llm
        self.max_repairs = max_repairs
        self.linker = SchemaLinker() if use_linking else None

    def schema_text(self, db_id: str, question: str, evidence: str = "") -> str:
        schema = load_schema(db_id)
        if self.linker is None:
            return schema.render()
        return schema.render(self.linker.select(schema, f"{question} {evidence}"))

    def ask(self, question: str, db_id: str, evidence: str = "") -> AgentResult:
        messages = initial_messages(question, evidence, self.schema_text(db_id, question, evidence))
        steps: list[Step] = []
        best: tuple[Step, ExecResult | None] | None = None
        for attempt in range(self.max_repairs + 1):
            reply = self.llm.generate_batch([messages])[0]
            step, result = check(db_id, extract_sql(reply), attempt)
            steps.append(step)
            if best is None or step.status in {"ok", "empty"}:
                best = (step, result)
            if not needs_repair(step):
                break
            messages = repair_messages(messages, reply, step)
        final_step, final_result = best
        return AgentResult(
            question=question,
            db_id=db_id,
            sql=final_step.sql,
            status=final_step.status,
            columns=final_result.columns if final_result else [],
            rows=final_result.rows if final_result else [],
            steps=steps,
        )
