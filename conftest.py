"""conftest.py — adds the project root to sys.path for all tests."""
import sys
from pathlib import Path

# Make sure `ingestion`, `extraction`, etc. are importable without install
sys.path.insert(0, str(Path(__file__).parent))
