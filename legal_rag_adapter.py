"""Wrap YOUR existing legal RAG so RAG Guard can call it.

RAG Guard only needs one endpoint:
    POST /query  {"question": "..."}  ->  {"answer": "...", "contexts": ["chunk 1", "chunk 2", ...]}

Edit the two lines marked EDIT, then run:
    uvicorn examples.legal_rag_adapter:app --port 8001
"""
from typing import List

from fastapi import FastAPI
from pydantic import BaseModel

# EDIT 1: import your own retrieval + generation functions
# from my_legal_rag.pipeline import retrieve, generate

app = FastAPI(title="Legal RAG adapter")


class Query(BaseModel):
    question: str


class RagAnswer(BaseModel):
    answer: str
    contexts: List[str]


@app.post("/query", response_model=RagAnswer)
def query(q: Query):
    # EDIT 2: replace this stub with your pipeline, e.g.
    # docs = retrieve(q.question, k=5)                 # list of LangChain Documents
    # contexts = [d.page_content for d in docs]
    # answer = generate(q.question, contexts)
    contexts = ["Section 420 IPC: Whoever cheats and thereby dishonestly induces the person deceived "
                "to deliver any property ... shall be punished with imprisonment of either description "
                "for a term which may extend to seven years, and shall also be liable to fine."]
    answer = ("Section 420 IPC deals with cheating and dishonestly inducing delivery of property. "
              "The punishment is imprisonment up to seven years and a fine. "
              "The offence is bailable in all cases.")   # last sentence is a deliberate hallucination
    return RagAnswer(answer=answer, contexts=contexts)
