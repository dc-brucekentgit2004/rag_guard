"""All settings in one place. Override any of them with environment variables."""
import os
from dataclasses import dataclass

try:  # read a .env file if python-dotenv is installed
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


@dataclass
class Settings:
    # NLI model that judges "does the source support this claim?"
    # base = fast (Colab T4 / CPU OK). For better accuracy try:
    # MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli
    nli_model: str = os.getenv("NLI_MODEL", "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli")

    # Any OpenAI-compatible endpoint. Examples:
    #   Groq   : https://api.groq.com/openai/v1
    #   Ollama : http://localhost:11434/v1   (api key can be anything)
    #   OpenAI : https://api.openai.com/v1
    llm_base_url: str = os.getenv("LLM_BASE_URL", "http://localhost:11434/v1")
    llm_api_key: str = os.getenv("LLM_API_KEY", "ollama")
    llm_model: str = os.getenv("LLM_MODEL", "llama3.1:8b")

    # Small open model used only to read token probabilities (uncertainty method)
    uq_model: str = os.getenv("UQ_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")

    # Guardrail policy
    support_threshold: float = float(os.getenv("SUPPORT_THRESHOLD", "0.5"))
    block_ratio: float = float(os.getenv("BLOCK_RATIO", "0.5"))
    on_fail: str = os.getenv("ON_FAIL", "rewrite")  # "rewrite" or "block"

    # Your existing RAG service (used by POST /ask)
    rag_url: str = os.getenv("RAG_URL", "http://localhost:8001/query")


settings = Settings()
