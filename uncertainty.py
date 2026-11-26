"""Method 3: Token-level uncertainty.

We feed (prompt + answer) into a small open LLM and read how probable it found
each answer token. Tokens the model was unsure about (low probability, high
entropy) are more likely to be invented. A claim's score is built from the
tokens that fall inside its character span.

Note: we score answers with a *proxy* model (e.g. Qwen2.5-0.5B), not the model
that wrote them. That is standard when the original model's log-probs are not
available, but say it clearly in your report.
"""
from typing import Dict, List, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .claims import Claim
from .config import settings


class TokenUncertainty:
    def __init__(self, model_name: str = None, device: Optional[str] = None, max_ctx: int = 2048):
        model_name = model_name or settings.uq_model
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        dtype = torch.float16 if self.device == "cuda" else torch.float32
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        try:  # transformers >= 4.56 uses `dtype`, older versions use `torch_dtype`
            self.model = AutoModelForCausalLM.from_pretrained(model_name, dtype=dtype)
        except TypeError:
            self.model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype)
        self.model.to(self.device).eval()
        self.max_ctx = max_ctx

    @torch.no_grad()
    def token_stats(self, prompt: str, response: str):
        """Per answer token: log-probability, entropy, and its (start, end) chars in `response`."""
        prompt_text = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True)
        p_ids = self.tokenizer(prompt_text, add_special_tokens=False).input_ids
        enc = self.tokenizer(response, add_special_tokens=False, return_offsets_mapping=True)
        r_ids, offsets = enc.input_ids, enc.offset_mapping

        # Keep the whole answer; cut the prompt from the left if too long.
        budget = self.max_ctx - len(r_ids)
        p_ids = p_ids[-budget:] if budget > 0 else p_ids[-1:]
        ids = torch.tensor([p_ids + r_ids], device=self.device)

        logits = self.model(ids).logits[0]
        start = len(p_ids)
        # logits at position i predict token i+1, so shift by one.
        logp = logits[start - 1:-1].float().log_softmax(dim=-1)
        target = ids[0, start:]
        token_logp = logp.gather(1, target[:, None]).squeeze(1)
        entropy = -(logp.exp() * logp).sum(dim=-1)
        return token_logp.cpu().tolist(), entropy.cpu().tolist(), offsets

    def score_claims(self, claims: List[Claim], prompt: str, response: str) -> List[Dict]:
        if not claims:
            return []
        logps, ents, offsets = self.token_stats(prompt, response)
        results = []
        for c in claims:
            idx = [i for i, (s, e) in enumerate(offsets) if s < c.end and e > c.start]
            if not idx:
                results.append({"hallucination_score": 0.0, "mean_nll": 0.0, "max_nll": 0.0,
                                "mean_entropy": 0.0})
                continue
            nll = [-logps[i] for i in idx]
            ent = [ents[i] for i in idx]
            results.append({
                "mean_nll": sum(nll) / len(nll),
                "max_nll": max(nll),             # the single most "surprising" token
                "mean_entropy": sum(ent) / len(ent),
                "hallucination_score": max(nll),  # used for ranking / AUROC
            })
        return results
