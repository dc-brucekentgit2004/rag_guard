"""Fast tests that need no model download: run with  pytest -q"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag_guard.claims import split_claims  # noqa: E402
from rag_guard.guardrail import Guardrail  # noqa: E402


def test_split_keeps_offsets():
    text = "Sec. 420 IPC covers cheating. The punishment is up to seven years.\n- It is non-bailable."
    claims = split_claims(text)
    assert [c.text for c in claims] == [
        "Sec. 420 IPC covers cheating.",
        "The punishment is up to seven years.",
        "It is non-bailable.",
    ]
    for c in claims:
        assert text[c.start:c.end] == c.text


class FakeNLI:
    """Pretends a claim is supported if all its words appear in the context."""
    def score_claims(self, claims, contexts):
        ctx = " ".join(contexts).lower()
        out = []
        for c in claims:
            ok = all(w.strip(".,").lower() in ctx for w in c.split())
            out.append({"entailment": 0.9 if ok else 0.1, "contradiction": 0.05, "evidence": ctx[:50]})
        return out


def test_decisions():
    g = Guardrail(nli=FakeNLI(), llm=None, support_threshold=0.5, block_ratio=0.5, on_fail="block")
    ctx = ["the punishment is up to seven years and a fine for cheating"]
    assert g.check("The punishment is up to seven years.", ctx)["decision"] == "pass"
    rep = g.check("The punishment is up to seven years. A fine for cheating. "
                  "It is bailable everywhere.", ctx)
    assert rep["decision"] == "flag" and "[unverified]" in rep["final_answer"]
    assert g.check("It is bailable everywhere. Courts love it always.", ctx)["decision"] == "block"


def test_nli_picks_best_chunk():
    import torch
    from rag_guard.nli import NLIScorer
    nli = NLIScorer.__new__(NLIScorer)          # skip model loading
    nli.ent_id, nli.con_id = 0, 2

    def fake_pair_probs(premises, hypotheses):
        # entailment high only when the claim's first word is in the chunk
        rows = [[0.9, 0.05, 0.05] if h.split()[0].lower() in p.lower() else [0.1, 0.3, 0.6]
                for p, h in zip(premises, hypotheses)]
        return torch.tensor(rows)
    nli.pair_probs = fake_pair_probs
    out = nli.score_claims(["Fine is mandatory here.", "Bail is automatic always."],
                           ["a fine is payable", "nothing else"])
    assert abs(out[0]["entailment"] - 0.9) < 1e-5 and out[0]["evidence"] == "a fine is payable"
    assert out[1]["entailment"] < 0.5 and out[1]["contradiction"] > 0.5
