"""Step 1 of the guardrail: break an answer into small, checkable claims.

Each claim keeps its character offsets (start, end) in the original answer,
so we can highlight it in the UI and compare it with RAGTruth's labelled spans.
"""
import re
from dataclasses import dataclass
from typing import List

# Abbreviations that end with "." but do NOT end a sentence.
_ABBREVIATIONS = (
    "e.g.", "i.e.", "etc.", "vs.", "Dr.", "Mr.", "Mrs.", "Ms.", "No.",
    "Sec.", "Art.", "Cl.", "Ltd.", "Pvt.", "Inc.", "Hon'ble.", "v.", "u/s.", "Rs.", "Re.", "Dr.", "St.", "Govt.", "Rs.", "Re.", "Dr.", "St.", "Govt.",
)

# A sentence ends at . ! or ? followed by whitespace and a capital/digit/quote/bracket,
# or at one or more newlines (bullet lists, paragraphs).
_BOUNDARY = re.compile(r"[.!?](?=\s+[A-Z0-9\"'(\[])|\n+")

# Leading list markers such as "- ", "* ", "1. ", "2) "
_LIST_MARKER = re.compile(r"^\s*(?:[-*•]|\d+[.)])?\s*")


@dataclass
class Claim:
    text: str
    start: int  # char offset in the original answer
    end: int


def _add(claims: List[Claim], text: str, s: int, e: int, min_words: int) -> None:
    segment = text[s:e]
    lead = _LIST_MARKER.match(segment).end()
    body = segment[lead:].rstrip()
    if len(body.split()) >= min_words:
        claims.append(Claim(text=body, start=s + lead, end=s + lead + len(body)))


def split_claims(text: str, min_words: int = 3) -> List[Claim]:
    """Rule-based sentence splitter that keeps offsets. Fast, no model needed."""
    claims: List[Claim] = []
    start = 0
    for m in _BOUNDARY.finditer(text):
        end = m.end()
        if m.group().startswith(".") and text[:end].endswith(_ABBREVIATIONS):
            continue  # "Sec. 420" is not a sentence end
        _add(claims, text, start, end, min_words)
        start = end
    _add(claims, text, start, len(text), min_words)
    return claims


DECOMPOSE_PROMPT = """Break the text below into short, self-contained factual claims.
Rules: one fact per line, replace pronouns with the names they refer to,
do not add anything that is not in the text, no numbering.

Text:
{text}

Claims:"""


def decompose_with_llm(text: str, llm) -> List[str]:
    """Optional: finer 'atomic facts' using an LLM (no offsets). Good for long sentences."""
    out = llm.ask(DECOMPOSE_PROMPT.format(text=text), temperature=0.0)
    lines = [l.strip(" -*•\t") for l in out.splitlines()]
    return [l for l in lines if len(l.split()) >= 3]
