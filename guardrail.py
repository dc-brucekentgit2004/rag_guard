"""The guardrail: split -> score -> decide -> (flag | block | rewrite).

Decision policy
  pass      every claim is supported by the sources
  flag      a few claims are unsupported (below block_ratio) and none is contradicted
            -> answer is shown with [unverified] markers
  corrected answer was rewritten from the sources and the rewrite passed the check
  block     too many unsupported / any contradicted claim, and rewrite failed or is off
"""
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional

from .claims import split_claims
from .config import settings

BLOCK_MESSAGE = ("I couldn't verify this answer against the source documents, "
                 "so I'm not showing it. Please rephrase the question or check the sources directly.")

REWRITE_PROMPT = """You are fixing an answer so that it only states facts found in the sources.

Sources:
{sources}

Question: {question}

Original answer:
{answer}

These statements were NOT supported by the sources:
{bad}

Rewrite the answer using only information in the sources. Remove or correct the
unsupported statements. If the sources do not answer the question, say so.
Answer:"""


@dataclass
class ClaimResult:
    text: str
    start: int
    end: int
    label: str            # supported | unverifiable | contradicted
    entailment: float
    contradiction: float
    evidence: str


class Guardrail:
    def __init__(self, nli, llm=None, support_threshold: float = None,
                 block_ratio: float = None, on_fail: str = None):
        self.nli = nli
        self.llm = llm
        self.support_threshold = support_threshold or settings.support_threshold
        self.block_ratio = block_ratio or settings.block_ratio
        self.on_fail = on_fail or settings.on_fail

    # ---------- scoring ----------
    def _label(self, ent: float, con: float) -> str:
        if ent >= self.support_threshold:
            return "supported"
        if con >= self.support_threshold:
            return "contradicted"
        return "unverifiable"

    def score(self, answer: str, contexts: List[str]) -> List[ClaimResult]:
        claims = split_claims(answer)
        scores = self.nli.score_claims([c.text for c in claims], contexts)
        return [
            ClaimResult(c.text, c.start, c.end, self._label(s["entailment"], s["contradiction"]),
                        round(s["entailment"], 4), round(s["contradiction"], 4), s["evidence"])
            for c, s in zip(claims, scores)
        ]

    # ---------- actions ----------
    @staticmethod
    def annotate(answer: str, results: List[ClaimResult]) -> str:
        """Insert [unverified] after each unsupported claim (work right-to-left so offsets stay valid)."""
        out = answer
        for r in sorted(results, key=lambda r: r.end, reverse=True):
            if r.label != "supported":
                tag = " [contradicts sources]" if r.label == "contradicted" else " [unverified]"
                out = out[:r.end] + tag + out[r.end:]
        return out

    def rewrite(self, question: str, answer: str, contexts: List[str],
                results: List[ClaimResult]) -> str:
        bad = "\n".join(f"- {r.text}" for r in results if r.label != "supported")
        sources = "\n\n".join(f"[{i + 1}] {c}" for i, c in enumerate(contexts))
        return self.llm.ask(REWRITE_PROMPT.format(
            sources=sources, question=question, answer=answer, bad=bad), temperature=0.0)

    # ---------- main entry point ----------
    def check(self, answer: str, contexts: List[str], question: str = "") -> Dict:
        results = self.score(answer, contexts)
        n = len(results)
        n_bad = sum(r.label != "supported" for r in results)
        n_con = sum(r.label == "contradicted" for r in results)
        faithfulness = 1.0 if n == 0 else (n - n_bad) / n

        if n_bad == 0:
            decision, final = "pass", answer
        elif n_bad / n < self.block_ratio and n_con == 0:
            decision, final = "flag", self.annotate(answer, results)
        elif self.on_fail == "rewrite" and self.llm is not None:
            try:
                new_answer = self.rewrite(question, answer, contexts, results)
            except Exception as e:  # LLM not running / wrong key: block instead of failing
                print(f"[guardrail] rewrite skipped, LLM unavailable: {type(e).__name__}: {e}")
                new_answer = None
            if new_answer:
                new_results = self.score(new_answer, contexts)
                if all(r.label == "supported" for r in new_results):
                    decision, final = "corrected", new_answer
                else:
                    decision, final = "block", BLOCK_MESSAGE
            else:
                decision, final = "block", BLOCK_MESSAGE
        else:
            decision, final = "block", BLOCK_MESSAGE

        return {
            "decision": decision,
            "faithfulness": round(faithfulness, 4),
            "n_claims": n,
            "n_unsupported": n_bad,
            "final_answer": final,
            "claims": [asdict(r) for r in results],
        }
