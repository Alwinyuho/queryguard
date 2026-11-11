"""Gradio demo. Run with: python -m queryguard.ui"""

from __future__ import annotations

import gradio as gr
import pandas as pd

from . import config
from .agent import QueryAgent
from .llm import llm_from_env

agent = QueryAgent(llm_from_env())
DATABASES = config.available_databases()


def respond(question: str, db_id: str, evidence: str):
    question = (question or "").strip()[:1000]
    if not question:
        return "", "Please type a question.", pd.DataFrame()
    result = agent.ask(question, db_id, (evidence or "").strip()[:1000])
    trace = "\n".join(
        f"Attempt {s.attempt + 1}: {s.status}{f' - {s.detail}' if s.detail else ''}"
        for s in result.steps
    )
    table = pd.DataFrame(result.rows[:200], columns=result.columns or None)
    return result.sql, trace, table


with gr.Blocks(title="QueryGuard") as demo:
    gr.Markdown(
        "# QueryGuard\nAsk a question about a database. The agent writes SQL, "
        "checks it is read-only, runs it in a sandbox, and repairs it if it fails."
    )
    with gr.Row():
        database = gr.Dropdown(
            DATABASES, value=DATABASES[0] if DATABASES else None, label="Database"
        )
        question = gr.Textbox(label="Question", lines=2, scale=3)
    evidence = gr.Textbox(label="Optional hint (business definitions)")
    button = gr.Button("Ask", variant="primary")
    sql = gr.Code(label="SQL", language="sql")
    trace = gr.Textbox(label="Agent steps", lines=3)
    table = gr.Dataframe(label="Result")
    button.click(respond, [question, database, evidence], [sql, trace, table])

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7861)
