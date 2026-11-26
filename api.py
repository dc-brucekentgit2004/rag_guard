"""FastAPI middleware.

Run:  uvicorn rag_guard.api:app --port 8000
  POST /guard  -> check an answer you already have  {answer, contexts, question?}
  POST /ask    -> forwards the question to your RAG (RAG_URL), then guards its answer
  GET  /health
"""
import time
from contextlib import asynccontextmanager
from typing import List

import httpx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from .config import settings
from .guardrail import Guardrail
from .llm import LLM
from .nli import NLIScorer

state = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load models once at startup, not on every request.
    nli = NLIScorer()
    try:
        llm = LLM()
    except Exception:
        llm = None  # rewrite disabled if no LLM configured
    state["guard"] = Guardrail(nli=nli, llm=llm)
    yield
    state.clear()


app = FastAPI(title="RAG Guard", version="0.1", lifespan=lifespan)


class GuardRequest(BaseModel):
    answer: str
    contexts: List[str]
    question: str = ""


class AskRequest(BaseModel):
    question: str


@app.get("/health")
def health():
    return {"status": "ok", "nli_model": settings.nli_model}


@app.post("/guard")
def guard(req: GuardRequest):
    t0 = time.perf_counter()
    report = state["guard"].check(req.answer, req.contexts, req.question)
    report["latency_ms"] = round((time.perf_counter() - t0) * 1000, 1)
    return report


@app.post("/ask")
def ask(req: AskRequest):
    # Your RAG service must return JSON: {"answer": "...", "contexts": ["chunk1", "chunk2", ...]}
    try:
        r = httpx.post(settings.rag_url, json={"question": req.question}, timeout=120)
        r.raise_for_status()
        rag = r.json()
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"RAG service error: {e}")

    t0 = time.perf_counter()
    report = state["guard"].check(rag["answer"], rag["contexts"], req.question)
    report["guard_latency_ms"] = round((time.perf_counter() - t0) * 1000, 1)
    report["original_answer"] = rag["answer"]
    report["contexts"] = rag["contexts"]
    return report
