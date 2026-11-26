"""Method 2: Self-consistency (SelfCheckGPT-NLI style).

Idea: if the model really "knows" a fact from the sources, it will say it again
when asked several times. Made-up details change between samples.
We re-ask the same prompt N times at high temperature, then use the NLI model
to check whether each sample supports each claim of the original answer.
"""
from typing import Dict, List, Optional

from .llm import LLM
from .nli import NLIScorer


class SelfConsistencyChecker:
    def __init__(self, llm: LLM, nli: NLIScorer, n_samples: int = 5, temperature: float = 1.0):
        self.llm = llm
        self.nli = nli
        self.n_samples = n_samples
        self.temperature = temperature

    def sample(self, prompt: str) -> List[str]:
        return self.llm.chat([{"role": "user", "content": prompt}],
                             temperature=self.temperature, n=self.n_samples)

    def score_claims(self, claims: List[str], prompt: str,
                     samples: Optional[List[str]] = None) -> List[Dict]:
        samples = samples or self.sample(prompt)
        samples = [s for s in samples if s.strip()]
        if not claims or not samples:
            return [{"hallucination_score": 0.5, "n_samples": len(samples)} for _ in claims]

        premises = [s for _ in claims for s in samples]
        hypotheses = [c for c in claims for _ in samples]
        probs = self.nli.pair_probs(premises, hypotheses).view(len(claims), len(samples), -1)

        results = []
        for k in range(len(claims)):
            p_ent = probs[k, :, self.nli.ent_id]
            p_con = probs[k, :, self.nli.con_id]
            # SelfCheckGPT-NLI score: P(contradiction) normalised by (entail + contradict)
            score = (p_con / (p_ent + p_con + 1e-8)).mean()
            results.append({"hallucination_score": float(score), "n_samples": len(samples)})
        return results
