"""Central configuration for the Multi-Agent Arabic Feedback Analysis prototype.

All values can be overridden through environment variables (see `.env.example`).
No API keys are hard-coded: keys are only read from the environment.
"""
from __future__ import annotations

import os
from datetime import date
from pathlib import Path

try:  # python-dotenv is optional at runtime
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # pragma: no cover
    pass

BASE_DIR = Path(__file__).resolve().parent
KB_DIR = BASE_DIR / "knowledge_base"
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "outputs"

# --------------------------------------------------------------------------- #
# LLM configuration
# --------------------------------------------------------------------------- #
# LLM_PROVIDER: "anthropic", "openai", or "offline".
#   "offline" uses the transparent rule-based reasoner in heuristics.py, so the
#   whole framework (and all tests) can run without an API key.
#   If the provider is "anthropic"/"openai" but no key is set, the system
#   automatically falls back to "offline" and says so in the decision log.
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "anthropic").strip().lower()

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-opus-5-5")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "")  # set to a model your account supports

LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "8000"))
LLM_MAX_RETRIES = int(os.getenv("LLM_MAX_RETRIES", "3"))  # our own retry loop
LLM_RETRY_BASE_DELAY = float(os.getenv("LLM_RETRY_BASE_DELAY", "2.0"))  # seconds
LLM_TIMEOUT = float(os.getenv("LLM_TIMEOUT", "120"))  # seconds per request

# --------------------------------------------------------------------------- #
# Workflow configuration
# --------------------------------------------------------------------------- #
MAX_REVISIONS = int(os.getenv("MAX_REVISIONS", "2"))  # Agent1 <-> Agent2 loop cap
MAX_WORKFLOW_STEPS = 20  # hard safety guard for the state machine

# Reference date used by the SLA checker. Leave empty to use today's date.
# A fixed date makes scenario results reproducible.
_ref = os.getenv("SLA_REFERENCE_DATE", "").strip()
SLA_REFERENCE_DATE: date | None = date.fromisoformat(_ref) if _ref else None

# --------------------------------------------------------------------------- #
# Knowledge-base / rule files (editable without touching code)
# --------------------------------------------------------------------------- #
EGSQUAL_KB_PATH = KB_DIR / "egsqual_dimensions.json"
DIALECT_LEXICON_PATH = KB_DIR / "saudi_dialect_lexicon.json"
ESCALATION_RULES_PATH = KB_DIR / "escalation_rules.json"
COMPLAINT_HISTORY_PATH = KB_DIR / "complaint_history_synthetic.json"
SEVERITY_RULES_PATH = KB_DIR / "severity_priority_rules.json"

DEFAULT_DATASET_PATH = DATA_DIR / "reviews.csv"


def llm_settings_summary() -> dict:
    """Non-secret summary of the LLM configuration (safe to display)."""
    return {
        "provider_requested": LLM_PROVIDER,
        "anthropic_model": ANTHROPIC_MODEL,
        "anthropic_key_set": bool(ANTHROPIC_API_KEY),
        "openai_model": OPENAI_MODEL or "(not set)",
        "openai_key_set": bool(OPENAI_API_KEY),
        "max_revisions": MAX_REVISIONS,
    }
