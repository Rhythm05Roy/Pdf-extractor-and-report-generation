"""
LLM client abstraction layer.
Priority chain: OpenAI → Gemini → Ollama → Stub

Clients are initialized lazily and cached. Call reset_client() to force re-init
(e.g. after updating the API key in the sidebar).
"""
import json
import time
from typing import Optional, Protocol
from loguru import logger
import config


class LLMClient(Protocol):
    def generate(self, prompt: str, temperature: float = 0.2) -> str: ...
    def model_name(self) -> str: ...


# OpenAI client
class OpenAIClient:
    """OpenAI chat-completion client (gpt-4o-mini by default)."""

    def __init__(self, api_key: str, model: str = "gpt-4o-mini"):
        from openai import OpenAI
        self._client = OpenAI(api_key=api_key)
        self._model = model
        logger.info(f"OpenAI client initialized: {model}")

    def generate(self, prompt: str, temperature: float = 0.2) -> str:
        for attempt in range(3):
            try:
                response = self._client.chat.completions.create(
                    model=self._model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temperature,
                    response_format={"type": "json_object"},  # enforces JSON output
                )
                return response.choices[0].message.content
            except Exception as e:
                if attempt < 2:
                    wait = 2 ** attempt
                    logger.warning(f"OpenAI attempt {attempt+1} failed: {e}. Retrying in {wait}s...")
                    time.sleep(wait)
                else:
                    raise

    def model_name(self) -> str:
        return f"openai/{self._model}"


# Gemini client
class GeminiClient:
    """Google Gemini API client."""

    # Models to try in order if the configured one fails
    _FALLBACK_MODELS = [
        "gemini-2.0-flash",
        "gemini-1.5-flash",
        "gemini-1.5-flash-latest",
        "gemini-pro",
    ]

    def __init__(self, api_key: str, model: str = "gemini-2.0-flash"):
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        self._genai = genai
        self._model_name = model
        self._model = None
        self._init_model(model)

    def _init_model(self, model_name: str):
        """Try to initialise the requested model; fall back if not found."""
        try:
            self._model = self._genai.GenerativeModel(model_name)
            self._model_name = model_name
        except Exception as e:
            logger.warning(f"Gemini model '{model_name}' init failed: {e}")
            self._model = None

    def generate(self, prompt: str, temperature: float = 0.2) -> str:
        generation_config = self._genai.GenerationConfig(temperature=temperature)

        # Try the configured model first, then fall back
        models_to_try = [self._model_name] + [
            m for m in self._FALLBACK_MODELS if m != self._model_name
        ]

        last_error = None
        for model_name in models_to_try:
            try:
                model = self._genai.GenerativeModel(model_name)
                for attempt in range(3):
                    try:
                        response = model.generate_content(
                            prompt,
                            generation_config=generation_config,
                        )
                        self._model_name = model_name  # remember which one worked
                        return response.text
                    except Exception as e:
                        err_str = str(e)
                        if "404" in err_str or "not found" in err_str.lower():
                            # This model doesn't exist — try next
                            last_error = e
                            break
                        if attempt < 2:
                            wait = 2 ** attempt
                            logger.warning(f"Gemini attempt {attempt+1} failed: {e}. Retrying in {wait}s...")
                            time.sleep(wait)
                        else:
                            last_error = e
                            break
            except Exception as e:
                last_error = e
                continue

        raise RuntimeError(
            f"All Gemini models failed. Last error: {last_error}. "
            "Consider setting OPENAI_API_KEY instead."
        )

    def model_name(self) -> str:
        return f"gemini/{self._model_name}"


# Ollama client
class OllamaClient:
    """Ollama local LLM client (HTTP API)."""

    def __init__(self, model: str = "llama3", base_url: str = "http://localhost:11434"):
        import requests
        self._model = model
        self._base_url = base_url
        self._requests = requests
        logger.info(f"Ollama client initialized: {model} @ {base_url}")

    def generate(self, prompt: str, temperature: float = 0.2) -> str:
        response = self._requests.post(
            f"{self._base_url}/api/generate",
            json={
                "model": self._model,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": temperature},
            },
            timeout=120,
        )
        response.raise_for_status()
        return response.json().get("response", "")

    def model_name(self) -> str:
        return f"ollama/{self._model}"


# Stub client (no LLM — structural fallback only)
class StubClient:
    """
    Rule-based fallback when no LLM is configured.
    Returns a structurally valid JSON response so the pipeline never crashes.
    """

    def generate(self, prompt: str, temperature: float = 0.2) -> str:
        logger.warning("Using stub LLM — set OPENAI_API_KEY or GEMINI_API_KEY in .env for AI drafts")
        return json.dumps({
            "memo_subject": "Case Fact Summary (AI unavailable)",
            "executive_summary": (
                "No LLM is configured. Add OPENAI_API_KEY or GEMINI_API_KEY to .env "
                "to enable AI-powered draft generation."
            ),
            "parties": [],
            "material_facts": [
                {"fact": "Configure an API key to enable AI analysis.", "source": ""}
            ],
            "key_dates": [],
            "relevant_provisions": [],
            "open_issues": [
                "Set OPENAI_API_KEY=sk-... in .env (recommended)",
                "Or set GEMINI_API_KEY=AIza... in .env",
            ],
        })

    def model_name(self) -> str:
        return "stub/no-llm"


# Factory — module-level singleton
_client: Optional[LLMClient] = None


def get_llm_client() -> LLMClient:
    """
    Return the best available LLM client.
    Priority: OpenAI → Gemini → Ollama → Stub
    """
    global _client
    if _client is not None:
        return _client

    # 1. OpenAI
    openai_key = getattr(config, "OPENAI_API_KEY", "") or ""
    if openai_key and not openai_key.startswith("your_"):
        try:
            _client = OpenAIClient(openai_key, config.OPENAI_MODEL)
            return _client
        except Exception as e:
            logger.warning(f"OpenAI init failed: {e}")

    # 2. Gemini
    gemini_key = config.GEMINI_API_KEY or ""
    if gemini_key and not gemini_key.startswith("your_"):
        try:
            _client = GeminiClient(gemini_key, config.GEMINI_MODEL)
            return _client
        except Exception as e:
            logger.warning(f"Gemini init failed: {e}")

    # 3. Ollama
    if config.USE_OLLAMA_FALLBACK:
        try:
            _client = OllamaClient(config.OLLAMA_MODEL, config.OLLAMA_BASE_URL)
            return _client
        except Exception as e:
            logger.warning(f"Ollama init failed: {e}")

    # 4. Stub
    logger.warning(
        "No LLM configured. Add OPENAI_API_KEY or GEMINI_API_KEY to .env"
    )
    _client = StubClient()
    return _client


def reset_client() -> None:
    """Force re-initialisation on next get_llm_client() call."""
    global _client
    _client = None
