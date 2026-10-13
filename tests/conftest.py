import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pytest  # noqa: E402

from llm import LLMClient  # noqa: E402
from workflow import MultiAgentWorkflow  # noqa: E402


@pytest.fixture(scope="session")
def offline_workflow():
    return MultiAgentWorkflow(LLMClient(provider="offline"), max_revisions=2)


DATASET = ROOT / "data" / "reviews.csv"
