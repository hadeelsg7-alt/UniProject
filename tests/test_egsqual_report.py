"""EGSQUAL classification, evidence, insights and the executive PDF report."""
import io
import re

import pytest
from pypdf import PdfReader

import insights as INS
import report_pdf
import tools
from llm import LLMClient, StubLLM
from schemas import LLMAgent1Analysis, LLMImpact, LLMImpactAssessment, LLMIssue, LLMTicketText, ReviewInput
from scenarios import get_scenarios
from workflow import MultiAgentWorkflow, state_to_row

WF = MultiAgentWorkflow(LLMClient(provider="offline"))


def _run(text, rating=1, rid="T"):
    return WF.run(ReviewInput(review_id=rid, text=text, app_id="APP", app_name="Test app", rating=rating))


# ---------------------------------------------------------------- knowledge base and verification status
def test_egsqual_reference_never_claims_unverified_definitions():
    ref = tools.egsqual_reference()
    assert ref["citation"]["doi"] == "10.1016/j.elerap.2022.101182"
    assert ref["verification"]["full_text_accessed"] is False
    assert len(ref["egsqual_dimensions"]) == 7
    assert all(d["definition_status"] == "not verified" for d in ref["egsqual_dimensions"])
    kb = tools.egsqual_kb()
    for d in ref["egsqual_dimensions"]:  # every crosswalk target exists and points back
        for key in d["prototype_categories"]:
            assert kb[key]["egsqual_candidate"] == d["name"]


def test_kb_has_unmapped_and_real_arabic_examples():
    kb = tools.egsqual_kb()
    assert "UNMAPPED" in kb and kb["UNMAPPED"]["egsqual_candidate"] is None
    for k, v in kb.items():
        assert v["examples_ar"], k
        assert all(re.search(r"[؀-ۿ]", e["text"]) for e in v["examples_ar"])


def test_kb_search_returns_matched_words_and_never_guesses():
    hit = tools.egsqual_kb_search("التطبيق يعلق ولا يعمل")
    assert hit["best_dimension"] == "SYSTEM_AVAILABILITY" and hit["matched"]["SYSTEM_AVAILABILITY"]
    assert tools.egsqual_kb_search("كلام عام بدون تفاصيل")["best_dimension"] == "UNMAPPED"


# ---------------------------------------------------------------- Agent 1 classification and evidence
def test_each_issue_has_dimension_reason_evidence_and_egsqual():
    st = _run("التطبيق يعلق كل ما افتح صفحة المواعيد")
    [i] = st.agent1.issues
    assert i.dimension == "SYSTEM_AVAILABILITY" and i.egsqual_candidate == "Website functionality"
    assert "يعلق" in i.dimension_reason
    assert i.evidence_verified and i.evidence in st.review.text


def test_multiple_issues_carry_multiple_dimensions():
    sc = next(s for s in get_scenarios() if s.key == "S2")
    st = WF.run(sc.review)
    dims = {i.dimension for i in st.agent1.issues}
    assert len(st.agent1.issues) >= 2 and len(dims) >= 2
    assert all(i.dimension_reason for i in st.agent1.issues)


def test_unspecific_ambiguous_issue_is_unmapped_not_guessed():
    st = _run("لا يوجد مخطط لأغلب المشاريع", rating=2)
    assert st.agent1.is_ambiguous
    unm = [i for i in st.agent1.issues if i.dimension == "UNMAPPED"]
    assert unm and unm[0].egsqual_candidate is None and "Unmapped" in unm[0].dimension_reason


def test_llm_dimension_reason_is_kept():
    issue = LLMIssue(description="Payment fails", issue_type="Bug", dimension="FULFILLMENT", evidence_quote="الدفع لا يعمل",
                     dimension_reason="The payment step cannot be completed.", affected_service="Payment", confidence=0.9)
    stub = StubLLM({"LLMAgent1Analysis": [LLMAgent1Analysis(feedback_type="Bug", issues=[issue], is_ambiguous=False)],
                    "LLMImpactAssessment": [LLMImpactAssessment(assessments=[LLMImpact(
                        issue_id="I1", service_completion="BLOCKED", scope="INDIVIDUAL", workaround_mentioned=False,
                        impact_summary="Cannot pay", supporting_quote="الدفع لا يعمل", reasoning="stub")])],
                    "LLMTicketText": [LLMTicketText(title="Payment fails", affected_service="Payment", issue_description="x")]})
    st = MultiAgentWorkflow(stub).run(ReviewInput(review_id="L", text="الدفع لا يعمل في التطبيق", rating=1))
    [i] = st.agent1.issues
    assert i.dimension_reason == "The payment step cannot be completed." and i.egsqual_candidate == "Website functionality"


# ---------------------------------------------------------------- insights, report, consistency
@pytest.fixture(scope="module")
def states():
    texts = ["التطبيق يعلق كل ما افتح صفحة المواعيد", "ما قدرت اسجل دخول ولا يوصل رمز التحقق",
             "الدفع لا يعمل والرسوم انخصمت مرتين", "التطبيق ممتاز وسهل الاستخدام", "لا يوجد مخطط لأغلب المشاريع",
             "ياليت تضيفون خدمة تجديد الرخصة"]
    out = [_run(t, rid=f"R{k}") for k, t in enumerate(texts)]
    sc = next(s for s in get_scenarios() if s.key == "S8")
    out.append(WF.run(sc.review, reference_date=sc.reference_date))
    return out


def test_empty_results_disable_the_report():
    ins = INS.compute_insights([])
    assert ins["coverage"]["reviews"] == 0 and ins["summary"] == []
    with pytest.raises(ValueError):
        report_pdf.build_executive_report(ins)


def test_insights_match_the_dashboard_numbers(states):
    from ui.pages import issue_frame

    ins = INS.compute_insights(states)
    frame = issue_frame(states)
    assert ins["coverage"]["issues"] == len(frame)
    assert dict((p, n) for p, n, _ in ins["priority"] if n) == frame["priority"].value_counts().to_dict()
    assert dict((s, n) for s, n, _ in ins["severity"] if n) == frame["severity"].value_counts().to_dict()
    rows = [state_to_row(s) for s in states]
    assert dict((a, n) for a, n, _ in ins["actions"]) == {a: sum(r["action"] == a for r in rows) for a in {r["action"] for r in rows}}
    assert ins["escalation_candidates"] and ins["escalation_candidates"][0]["reference"] == "SYN-2026-0417"
    assert sum(d["issues"] for d in ins["dimensions"]) == ins["coverage"]["issues"]
    for r in ins["recommendations"]:  # every recommendation states a numeric basis
        assert re.search(r"\d", r["basis"])


def test_pdf_report_sections_numbers_label_and_arabic(states):
    ins = INS.compute_insights(states)
    data = report_pdf.build_executive_report(ins)
    assert data[:4] == b"%PDF"
    reader = PdfReader(io.BytesIO(data))
    text = "\n".join(p.extract_text() for p in reader.pages)
    for title in ["Executive Summary", "Dataset and Analysis Coverage", "Key Service-Quality Findings", "Top Five Reported Issues",
                  "Priority Distribution", "Most Affected EGSQUAL Dimensions", "Recommended Improvement Actions",
                  "Simulated Escalation Summary", "Management Recommendations", "Research Limitations"]:
        assert title in text, title
    assert INS.PROTOTYPE_LABEL in text and "BeneficiaryVoice AI" in text
    assert 2 <= len(reader.pages) <= 4  # concise executive report
    cov = ins["coverage"]
    assert f"{cov['reviews']} beneficiary reviews" in text  # same number as the page
    assert re.search(r"[؀-ۿﭐ-﻿]", text)  # Arabic evidence is embedded as real text
    assert b"DejaVuSans" in data  # the bundled Arabic-capable font is embedded


def test_rtl_detection():
    assert report_pdf.is_arabic("التطبيق يعلق كل ما افتح")
    assert not report_pdf.is_arabic("Payment / fees: failure reported")
    assert report_pdf.is_arabic("R0001 التطبيق لا يعمل")


def _exec_script():
    from datetime import date

    import streamlit as st

    import tools
    from llm import LLMClient
    from ui import pages as P
    from workflow import MultiAgentWorkflow

    llm = LLMClient(provider="offline")
    st.session_state["ctx"] = P.Ctx(llm=llm, workflow=MultiAgentWorkflow(llm), df=None, info=None, ref_date=date(2026, 1, 1),
                                    matrix=tools.priority_matrix_kb()["matrix"], sla_targets=tools.sla_targets_kb()["targets"])
    P.page_executive()


def test_executive_dashboard_generates_the_report(states):
    """The Executive Dashboard (with the merged insights) renders and builds the PDF from real results."""
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_function(_exec_script)
    at.session_state["store"] = {f"t:{s.review.review_id}": s for s in states}
    at.run(timeout=60)
    assert not at.exception
    at.button(key="ins_gen").click().run(timeout=60)
    assert not at.exception and "ins_pdf" in at.session_state
    assert "Not generated" in " ".join(c.value for c in at.caption)  # 7 reviews: no management recommendations


def test_executive_dashboard_offers_no_report_without_results():
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_function(_exec_script)
    at.run(timeout=60)
    assert not at.exception
    assert not [b for b in at.button if b.key == "ins_gen"]  # nothing to report: no button, an explanation instead


def test_small_samples_get_no_management_recommendations(states):
    ins = INS.compute_insights(states)
    assert not ins["sample_ok"] and ins["recommendations"] == []
    assert any("Management recommendations need at least" in line for line in ins["summary"])
    text = "\n".join(p.extract_text() for p in PdfReader(io.BytesIO(report_pdf.build_executive_report(ins))).pages)
    assert "Not generated" in text
