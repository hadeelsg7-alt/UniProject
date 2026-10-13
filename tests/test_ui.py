"""UI tests: the dashboard runs, and the visual components stay faithful to the real execution."""
import pytest

from llm import LLMClient, LLMError, StubLLM
from scenarios import get_scenarios
from schemas import ReviewInput
from ui import components as C
from workflow import MultiAgentWorkflow


@pytest.fixture
def captured(monkeypatch):
    out = []
    monkeypatch.setattr(C, "md", lambda html: out.append(html))
    return out


def _s6():
    sc = next(s for s in get_scenarios() if s.key == "S6")
    return MultiAgentWorkflow(LLMClient(provider="offline")).run(sc.review)


def test_replay_never_shows_future_results(captured):
    st = _s6()
    assert st.node_trace[2] == "A1_REVIEW_RESPONSE"
    C.workflow_diagram(st, step=2)          # Agent 1 answering the review request
    html = captured[-1]
    assert "provisional" in html and "VERIFIED_AFTER_REVISION" not in html
    assert "1 / 2 round(s)" in html and "TYPE_CONFLICT" in html
    assert "Waiting for Agent 2" in html     # Agent 3 has not run yet at this step
    C.workflow_diagram(st)                   # finished run
    assert "VERIFIED_AFTER_REVISION" in captured[-1]


def test_no_revision_shown_when_none_happened(captured):
    sc = next(s for s in get_scenarios() if s.key == "S1")
    st = MultiAgentWorkflow(LLMClient(provider="offline")).run(sc.review)
    C.workflow_diagram(st)
    assert "no revision was requested" in captured[-1]


def test_fallback_is_never_labelled_as_llm(captured):
    stub = StubLLM({"LLMAgent1Analysis": [LLMError("simulated outage")]})
    st = MultiAgentWorkflow(stub).run(ReviewInput(review_id="x", text="التطبيق لا يعمل ويطلع خطأ", rating=1))
    label, kind = C.agent_reasoner(st, "Agent1", stub)
    assert label.startswith("Fallback") and kind == "bad"
    C.fallback_banner(st, stub)
    assert "LLM failure" in captured[-1] and "NOT produced by the LLM" in captured[-1]
    offline = LLMClient(provider="offline")
    assert C.agent_reasoner(st, "Agent1", offline)[0] == "Rule-based (offline)"


def test_user_text_is_escaped(captured):
    st = MultiAgentWorkflow(LLMClient(provider="offline")).run(
        ReviewInput(review_id="x", text='<script>alert(1)</script> التطبيق لا يعمل', rating=1))
    C.review_with_evidence(st.review.text, st.agent1.issues)
    C.workflow_diagram(st)
    assert all("<script>" not in h for h in captured)


def test_dashboard_runs_sample_analysis():
    from streamlit.testing.v1 import AppTest
    from conftest import ROOT
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=180).run()
    assert not at.exception
    btn = [b for b in at.button if "Run sample analysis" in b.label]
    if btn and not btn[0].disabled:  # needs data/reviews.csv
        btn[0].click().run()
        assert not at.exception
        assert any("Issues extracted" in m.value.replace("ISSUES", "Issues") or "Issues extracted" in m.value for m in at.markdown)
