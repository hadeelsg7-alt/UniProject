"""Headless rendering of all seven pages (empty and with analysed results) and of app.py itself."""
import pytest
from streamlit.testing.v1 import AppTest

from llm import LLMClient
from scenarios import get_scenarios
from schemas import ReviewInput
from workflow import MultiAgentWorkflow

PAGES = ["page_data_run", "page_agent1", "page_agent2", "page_agent3", "page_executive", "page_eval_dashboard",
         "page_methodology", "page_settings"]


def _script(name):
    from datetime import date

    import streamlit as st

    import tools
    from llm import LLMClient
    from ui import agent_pages as A
    from ui import eval_dashboard as E
    from ui import pages as P
    from workflow import MultiAgentWorkflow

    llm = LLMClient(provider="offline")
    st.session_state["ctx"] = P.Ctx(llm=llm, workflow=MultiAgentWorkflow(llm), df=None, info=None, ref_date=date(2026, 1, 1),
                                    matrix=tools.priority_matrix_kb()["matrix"], sla_targets=tools.sla_targets_kb()["targets"])
    getattr(next(m for m in (P, A, E) if hasattr(m, name)), name)()


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
    at.session_state["current_rid"] = list(store.values())[-1].review.review_id  # the S6 review, with a revision loop
    at.run(timeout=60)
    assert not at.exception
    md = " ".join(m.value for m in at.markdown)
    assert "Impact" in md and "Urgency" in md
    assert at.selectbox(key="rid_a2").value == list(store.values())[-1].review.review_id


def test_full_app_starts_on_the_executive_dashboard():
    at = AppTest.from_file("../app.py")
    at.run(timeout=90)
    assert not at.exception, at.exception


def test_data_page_runs_the_workflow_and_shares_results():
    """Data & Run Analysis is the only place that starts the analysis; results then appear on the agent pages."""
    def script():
        from datetime import date

        import streamlit as st

        import tools
        from data_loader import load_reviews
        from llm import LLMClient
        from ui import pages as P
        from workflow import MultiAgentWorkflow

        df, info = load_reviews("Abbreviation,Rating,Content\nAPP,1,التطبيق يعلق كل ما افتحه\nAPP,5,تطبيق ممتاز\n".encode("utf-8"))
        llm = LLMClient(provider="offline")
        st.session_state["ctx"] = P.Ctx(llm=llm, workflow=MultiAgentWorkflow(llm), df=df, info=info, ref_date=date(2026, 1, 1),
                                        matrix=tools.priority_matrix_kb()["matrix"], sla_targets=tools.sla_targets_kb()["targets"])
        P.page_data_run()

    at = AppTest.from_function(script)
    at.run(timeout=60)
    assert not at.exception
    at.number_input(key="run_n").set_value(2)
    at.button(key="run_go").click().run(timeout=60)
    assert not at.exception
    assert len(at.session_state["store"]) == 2
    assert at.session_state["last_run"]["Rule-based (offline)"] == 2 and at.session_state["last_run"]["Failed"] == 0
