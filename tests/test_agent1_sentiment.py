"""Agent 1 step 1 (sentiment), step 2-4 outputs, hand-over to Agent 2, and the logo."""
import pytest

import config
import tools
from llm import LLMClient, StubLLM
from schemas import (LLMAgent1Analysis, LLMImpact, LLMImpactAssessment, LLMIssue, LLMTicketText, ReviewInput)
from workflow import MultiAgentWorkflow

WF = MultiAgentWorkflow(LLMClient(provider="offline"))


def _run(text, rating=None, rid="S"):
    return WF.run(ReviewInput(review_id=rid, text=text, app_id="APP", app_name="Test app", rating=rating))


@pytest.mark.parametrize("text,expected", [
    ("التطبيق ممتاز وسهل الاستخدام الله يعطيكم العافية", "Positive"),
    ("التطبيق سيء جدا ويعلق كل ما افتحه", "Negative"),
    ("كيف اغير رقم الجوال في التطبيق؟", "Neutral"),
    ("تطبيق رائع وسهل لكن الدفع لا يعمل ويطلع خطأ", "Mixed"),
])
def test_agent1_sentiment_four_classes(text, expected):
    st = _run(text, rating=3)
    assert st.agent1.sentiment == expected
    assert st.agent1.sentiment_source == "lexicon"
    if expected != "Neutral":
        assert st.agent1.sentiment_evidence and all(e.split()[-1] in text for e in st.agent1.sentiment_evidence)


def test_negated_praise_is_negative_and_rating_does_not_change_the_label():
    r = tools.sentiment_analyzer("مو ممتاز ابد", rating=5)
    assert r["label"] == "Negative" and r["negative_cues"] == ["مو ممتاز"] and r["rating_indicator"] == "Positive"


def test_multiple_issues_get_their_own_dimension_and_evidence():
    st = _run("حجزت موعد وطلع تم الحجز بس لما رحت ما فيه موعد، وفوق كذا التطبيق يعلق كل ما افتح صفحة المواعيد", rating=1)
    assert len(st.agent1.issues) >= 2
    assert all(i.evidence in st.review.text and i.dimension and i.dimension_reason for i in st.agent1.issues)


def test_agent1_outputs_reach_agent2_and_sentiment_never_sets_severity():
    # mixed tone, 5 stars, but the service is blocked -> severity comes from impact, not tone or rating
    st = _run("تطبيق رائع لكن ما اقدر اسجل دخول ابد ولا يوصل رمز التحقق", rating=5)
    a1, a2 = st.agent1, st.agent2
    assert a1.sentiment == "Mixed" and a2.sentiment_intensity == a1.sentiment
    handoff = next(m for m in st.messages if m.sender == "Agent1" and m.receiver == "Agent2")
    first = st.agent1_history[0]  # the hand-over carries Agent 1's first analysis (before any critique round)
    for part in ("sentiment=Mixed", f"type={first.feedback_type}", "dimensions=", "ambiguous="):
        assert part in handoff.content
    assert any(e.agent == "Agent2" and e.action == "INPUT" and "sentiment Mixed" in e.detail for e in st.decision_log)
    assert a2.overall_severity is not None and a2.overall_severity.value in ("High", "Critical")
    # strongly negative 1-star text with no operational issue gets no severity at all
    st2 = _run("سيء جدا جدا فاشل", rating=1)
    assert st2.agent1.sentiment == "Negative" and st2.agent2.overall_severity is None


def test_llm_sentiment_is_used_and_its_evidence_verified():
    text = "تطبيق رائع لكن الدفع لا يعمل"
    issue = LLMIssue(description="Payment fails", issue_type="Bug", dimension="FULFILLMENT", evidence_quote="الدفع لا يعمل",
                     affected_service="Payment", confidence=0.9)
    stub = StubLLM({
        "LLMAgent1Analysis": [LLMAgent1Analysis(sentiment="Mixed", sentiment_evidence=["رائع", "كلام غير موجود"],
                                                feedback_type="Bug", issues=[issue], is_ambiguous=False)],
        "LLMImpactAssessment": [LLMImpactAssessment(assessments=[LLMImpact(
            issue_id="I1", service_completion="BLOCKED", scope="INDIVIDUAL", workaround_mentioned=False,
            impact_summary="Cannot pay", supporting_quote="الدفع لا يعمل", reasoning="stub")])],
        "LLMTicketText": [LLMTicketText(title="Payment fails", affected_service="Payment", issue_description="x")],
    })
    st = MultiAgentWorkflow(stub).run(ReviewInput(review_id="L", text=text, rating=4))
    assert st.agent1.sentiment == "Mixed" and st.agent1.sentiment_source == "LLM"
    assert st.agent1.sentiment_evidence == ["رائع"]  # the invented span is dropped


def test_logo_is_bundled_and_used_in_pdf_and_pages():
    import insights as INS
    import report_pdf
    from ui import components as C

    assert config.LOGO_PATH.exists()
    assert C.logo_data_uri().startswith("data:image/png;base64,")
    st = _run("التطبيق يعلق كل ما افتحه", rating=1)
    pdf = report_pdf.build_executive_report(INS.compute_insights([st]))
    assert b"/Subtype /Image" in pdf
