"""
Central configuration for LegalMind.
All settings read from environment variables with sensible defaults.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
SAMPLE_INPUTS_DIR = DATA_DIR / "sample_inputs"
SAMPLE_OUTPUTS_DIR = BASE_DIR / "sample_outputs"

# OpenAI (priority 1)
OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

# Gemini (priority 2) — use gemini-2.0-flash as default (widely available)
GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL: str = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")

# Ollama local fallback (priority 3)
USE_OLLAMA_FALLBACK: bool = os.getenv("USE_OLLAMA_FALLBACK", "false").lower() == "true"
OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "llama3")
OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

TESSERACT_CMD: str = os.getenv("TESSERACT_CMD", "/usr/bin/tesseract")

# Mistral (OCR + optional LLM)
MISTRAL_API_KEY: str = os.getenv("MISTRAL_API_KEY", os.getenv("mistral_api", ""))
MISTRAL_OCR_MODEL: str = os.getenv("MISTRAL_OCR_MODEL", "mistral-ocr-latest")
USE_MISTRAL_OCR: bool = os.getenv("USE_MISTRAL_OCR", "true").lower() == "true"

CHROMA_DB_PATH: str = os.getenv("CHROMA_DB_PATH", str(DATA_DIR / "chroma_db"))
RETRIEVAL_TOP_K: int = int(os.getenv("RETRIEVAL_TOP_K", "8"))
DENSE_WEIGHT: float = float(os.getenv("DENSE_WEIGHT", "0.6"))  # weight for vector similarity
BM25_WEIGHT: float = float(os.getenv("BM25_WEIGHT", "0.4"))    # weight for keyword score
EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "150"))

EDITS_DB_PATH: str = os.getenv("EDITS_DB_PATH", str(DATA_DIR / "edits.db"))
PATTERNS_FILE: str = os.getenv("PATTERNS_FILE", str(DATA_DIR / "edit_patterns.json"))

LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")

MAX_EVIDENCE_CHUNKS: int = int(os.getenv("MAX_EVIDENCE_CHUNKS", "6"))
MAX_PATTERNS_IN_PROMPT: int = int(os.getenv("MAX_PATTERNS_IN_PROMPT", "5"))
