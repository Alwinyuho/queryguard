"""FastAPI service: ask a question about one of the databases and get SQL plus results."""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from . import config


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=1000)
    db_id: str
    evidence: str = Field(default="", max_length=1000)


def create_app(agent=None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if agent is None:
            from .agent import QueryAgent
            from .llm import llm_from_env

            app.state.agent = QueryAgent(llm_from_env())
        else:
            app.state.agent = agent
        yield

    app = FastAPI(title="QueryGuard", version="0.1.0", lifespan=lifespan)

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok", "model": app.state.agent.llm.name}

    @app.get("/databases")
    def databases() -> list[str]:
        return config.available_databases()

    @app.post("/ask")
    def ask(request: AskRequest) -> dict:
        if request.db_id not in config.available_databases():
            raise HTTPException(status_code=404, detail=f"unknown database: {request.db_id}")
        question = request.question.strip()
        if not question:
            raise HTTPException(status_code=422, detail="question is blank")
        return app.state.agent.ask(question, request.db_id, request.evidence.strip()).to_dict()

    return app


app = create_app()
