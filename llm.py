"""Thin wrapper around any OpenAI-compatible chat API (Groq, Ollama, OpenAI, vLLM...)."""
from typing import Dict, List

from openai import OpenAI

from .config import settings


class LLM:
    def __init__(self, base_url: str = None, api_key: str = None, model: str = None):
        self.client = OpenAI(
            base_url=base_url or settings.llm_base_url,
            api_key=api_key or settings.llm_api_key or "not-needed",
            timeout=60, max_retries=0,   # fail fast instead of hanging if the LLM is not running
        )
        self.model = model or settings.llm_model

    def chat(self, messages: List[Dict], temperature: float = 0.0,
             n: int = 1, max_tokens: int = 512) -> List[str]:
        # Many providers ignore n>1, so we loop instead.
        outputs = []
        for _ in range(n):
            resp = self.client.chat.completions.create(
                model=self.model, messages=messages,
                temperature=temperature, max_tokens=max_tokens,
            )
            outputs.append(resp.choices[0].message.content or "")
        return outputs

    def ask(self, prompt: str, **kwargs) -> str:
        return self.chat([{"role": "user", "content": prompt}], **kwargs)[0]
