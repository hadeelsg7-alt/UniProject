"""Headless rendering of all seven pages (empty and with analysed results) and of app.py itself."""
import pytest
from streamlit.testing.v1 import AppTest

from llm import LLMClient
from scenarios import get_scenarios
from schemas import ReviewInput
from workflow import MultiAgentWorkflow

PAGES = ["page_executive", "page_agent1", "page_agent2", "page_agent3", "page_results", "page_insights", "page_methodology"]


def _script(name):
    from datetime import date

    import streamlit as st

    import tools
    from llm import LLMClient
    from ui import agent_pages as A
    from ui import pages as P
    from workflow import MultiAgentWorkflow

    llm = LLMClient(provider="offline")
    st.session_state["ctx"] = P.Ctx(llm=llm, workflow=MultiAgentWorkflow(llm), df=None, info=None, ref_date=date(2026, 1, 1),
                                    matrix=tools.priority_matrix_kb()["matrix"], sla_targets=tools.sla_targets_kb()["targets"])
    getattr(P if hasattr(P, name) else A, name)()


@pytest.fixture(scope="module")
def store():
    wf = MultiAgentWorkflow(LLMClient(provider="offline"))
    states = [wf.run(ReviewInput(review_id=f"R{k}", text=t, app_id="APP", app_name="Test app", rating=1))
              for k, t in enumerate(["الدفع لا يعمل والمبلغ انخصم", "التطبيق يعلق كل ما افتح المواعيد", "تطبيق ممتاز"])]
    s6 = next(s for s in get_scenarios() if s.key == "S6")
    states.append(wf.run(s6.review))
    return {f"t:{s.review.review_id}": s for s in states}


@pytest.mark.parametrize("name", PAGES)
def test_page_renders_without_results(name):
    at = AppTest.from_function(_script, args=(name,))
    at.run(timeout=60)
    assert not at.exception, at.exception


@pytest.mark.parametrize("name", PAGES)
def test_page_renders_with_results(name, store):
    at = AppTest.from_function(_script, args=(name,))
    at.session_state["store"] = store
    at.run(timeout=60)
    assert not at.exception, at.exception


def test_agent_pages_show_the_same_selected_review(store):
    at = AppTest.from_function(_script, args=("page_agent2",))
    at.session_state["store"] = store
    at.session_state["current_key"] = list(store)[-1]  # the S6 review, which has a revision loop
    at.run(timeout=60)
    assert not at.exception
    md = " ".join(m.value for m in at.markdown)
    assert "Impact" in md and "Urgency" in md
    assert at.selectbox(key="pick_a2").value == list(store)[-1]


def test_full_app_starts_on_the_executive_dashboard():
    at = AppTest.from_file("../app.py")
    at.run(timeout=90)
    assert not at.exception, at.exception
