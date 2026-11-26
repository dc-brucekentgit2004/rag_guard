"""Method 1: NLI (Natural Language Inference) checker.

An NLI model reads a (premise, hypothesis) pair and outputs three probabilities:
entailment (premise supports it), neutral (not mentioned), contradiction.
Premise = a chunk of the retrieved source. Hypothesis = one claim from the answer.
A claim is "supported" if at least one source chunk entails it.
"""
from typing import Dict, List, Optional

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from .config import settings


class NLIScorer:
    def __init__(self, model_name: str = None, device: Optional[str] = None,
                 max_length: int = 512, batch_size: int = 16):
        model_name = model_name or settings.nli_model
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.to(self.device).eval()
        self.max_length = max_length
        self.batch_size = batch_size

        # Label order differs between models, so read it from the config.
        id2label = {i: l.lower() for i, l in self.model.config.id2label.items()}
        label2id = {l: i for i, l in id2label.items()}
        self.ent_id = label2id["entailment"]
        self.con_id = label2id["contradiction"]

    @staticmethod
    def chunk(text: str, chunk_words: int = 250, overlap: int = 50) -> List[str]:
        """NLI models only read ~512 tokens, so long sources are cut into overlapping windows."""
        words = text.split()
        if len(words) <= chunk_words:
            return [text]
        step = chunk_words - overlap
        chunks = []
        for i in range(0, len(words), step):
            chunks.append(" ".join(words[i:i + chunk_words]))
            if i + chunk_words >= len(words):
                break
        return chunks

    @torch.no_grad()
    def pair_probs(self, premises: List[str], hypotheses: List[str]) -> torch.Tensor:
        """Returns a (N, 3) tensor of label probabilities for N (premise, hypothesis) pairs."""
        all_probs = []
        for i in range(0, len(premises), self.batch_size):
            enc = self.tokenizer(
                premises[i:i + self.batch_size], hypotheses[i:i + self.batch_size],
                truncation="only_first", max_length=self.max_length,
                padding=True, return_tensors="pt",
            ).to(self.device)
            logits = self.model(**enc).logits
            all_probs.append(logits.softmax(dim=-1).cpu())
        return torch.cat(all_probs) if all_probs else torch.empty(0, 3)

    def score_claims(self, claims: List[str], contexts: List[str]) -> List[Dict]:
        """For every claim: best entailment over all source chunks + that chunk as evidence."""
        chunks = [c for ctx in contexts for c in self.chunk(ctx)]
        if not claims:
            return []
        if not chunks:
            return [{"entailment": 0.0, "contradiction": 0.0, "evidence": ""} for _ in claims]

        premises = [ch for _ in claims for ch in chunks]   # claim-major order
        hypotheses = [cl for cl in claims for _ in chunks]
        probs = self.pair_probs(premises, hypotheses).view(len(claims), len(chunks), -1)

        results = []
        for k in range(len(claims)):
            ent = probs[k, :, self.ent_id]
            con = probs[k, :, self.con_id]
            best = int(ent.argmax())
            results.append({
                "entailment": float(ent[best]),
                "contradiction": float(con.max()),
                "evidence": chunks[best],
            })
        return results
