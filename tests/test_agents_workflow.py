"""Functional tests of agent behaviour, inter-agent collaboration and the revision loop (offline reasoner)."""
from schemas import ActionType, AgentMessage, EscalationStatus, ReviewInput, VerificationStatus
from agents.agent1 import FeedbackInvestigationAgent
from schemas import WorkflowState
import pytest


def run(wf, text, rating=1, app="TEST"):
    return wf.run(ReviewInput(review_id="T1", text=text, rating=rating, app_id=app, app_name=app))


def test_agent1_extracts_multiple_issues_with_verified_evidence(offline_workflow):
    st = run(offline_workflow, "تطبيق فاشل يقفل كل شوي وتسجيل الدخول مشكله وتختار اللغة العربية لكن جميع الخيارات باللغة الإنجليزية")
    assert len(st.agent1.issues) >= 3
    assert all(i.evidence_verified for i in st.agent1.issues)
    assert all(i.evidence in st.review.text for i in st.agent1.issues)
    assert len({i.dimension for i in st.agent1.issues}) >= 2


def test_one_star_is_not_automatically_critical(offline_workflow):
    st = run(offline_workflow, "فاشلين في كل شي", rating=1)
    assert st.agent2.overall_severity is None
    assert st.agent1.is_ambiguous
    assert st.agent3.action == ActionType.REQUEST_ADDITIONAL_INFO


def test_low_severity_despite_negative_sentiment(offline_workflow):
    st = run(offline_workflow, "للاسف التصميم سيء والواجهة معقدة", rating=1)
    assert st.agent2.overall_severity is not None
    assert st.agent2.overall_severity.value in ("Low", "Medium")


def test_revision_loop_type_conflict(offline_workflow):
    st = run(offline_workflow, "برنامج رائع وللمعلومية عند إختيار اللغه العربيه التطبيق لا يقبل سوى الأرقام العربية", rating=5)
    reqs = [m for m in st.messages if m.message_type == "REVIEW_REQUEST"]
    resps = [m for m in st.messages if m.message_type == "REVIEW_RESPONSE"]
    assert reqs and resps and reqs[0].concern_code == "TYPE_CONFLICT"
    assert resps[0].content.startswith("REVISE")
    assert st.agent1_history[0].feedback_type == "Praise" and st.agent1.feedback_type == "Bug"
    assert st.agent2.verification_status == VerificationStatus.VERIFIED_AFTER_REVISION
    assert st.node_trace == ["A1_ANALYZE", "A2_ASSESS", "A1_REVIEW_RESPONSE", "A2_ASSESS", "A3_RESOLVE", "END"]


def test_revision_loop_confirm_then_insufficient_evidence(offline_workflow):
    st = run(offline_workflow, "لا يعمل")
    assert st.revision_count == 1
    assert any(m.content.startswith("CONFIRM") for m in st.messages if m.message_type == "REVIEW_RESPONSE")
    a = st.agent2.assessments[0]
    assert a.verification_status == VerificationStatus.INSUFFICIENT_EVIDENCE and a.provisional
    assert a.priority.value == "P3"  # capped by P-VERIFY
    assert st.agent3.action == ActionType.REQUEST_ADDITIONAL_INFO


def test_revision_loop_is_bounded(offline_workflow):
    from llm import LLMClient
    from workflow import MultiAgentWorkflow
    for k in (0, 1, 2):
        wf = MultiAgentWorkflow(LLMClient(provider="offline"), max_revisions=k)
        st = run(wf, "لا يعمل")
        assert st.revision_count <= k
        assert st.agent3 is not None
    st0 = run(MultiAgentWorkflow(LLMClient(provider="offline"), max_revisions=0), "لا يعمل")
    assert any(e.action == "LIMIT_REACHED" for e in st0.decision_log)


def test_praise_only_requires_no_action(offline_workflow):
    st = run(offline_workflow, "تطبيق رائع وسهل الاستخدام شكرا لكم", rating=5)
    assert st.agent1.feedback_type == "Praise" and not st.agent1.issues
    assert st.agent3.action == ActionType.NO_ACTION and st.agent3.ticket is None


def test_pending_verification_never_invents_complaint_data(offline_workflow):
    st = run(offline_workflow, "رفعت شكوى من اسبوعين ولا احد رد والتطبيق لا يقبل تسجيل الدخول")
    esc = st.agent3.escalation
    assert esc.status == EscalationStatus.PENDING_VERIFICATION
    assert esc.complaint_reference is None and esc.complaint_record is None and esc.sla_deadline is None
    assert {"complaint_reference", "complaint_status", "submission_date"} <= set(esc.missing_fields)
    assert st.agent3.human_approval_required


def test_ticket_has_required_fields_and_is_simulated(offline_workflow):
    import tools
    st = run(offline_workflow, "سددت الرسوم ولم يتم التسجيل")
    t = st.agent3.ticket
    assert t is not None and t.simulated and t.human_approval_required
    assert tools.ticket_fields_validator(t.model_dump())["complete"]
    assert st.agent2.overall_severity.value == "Critical"


def test_agents_cannot_use_other_agents_tools():
    agent = FeedbackInvestigationAgent()
    st = WorkflowState(review=ReviewInput(review_id="x", text="x"))
    with pytest.raises(PermissionError):
        agent.call_tool(st, "complaint_history_lookup", "should fail", reference="SYN-2026-0417")


def test_decision_log_and_tool_calls_recorded(offline_workflow):
    st = run(offline_workflow, "التطبيق يعلق ولا يفتح")
    agents_logged = {e.agent for e in st.decision_log}
    assert {"Agent1", "Agent2", "Agent3", "Orchestrator"} <= agents_logged
    assert {t.agent for t in st.tool_calls} == {"Agent1", "Agent2", "Agent3"}


def test_batch_survives_bad_input(offline_workflow):
    reviews = [ReviewInput(review_id=f"B{i}", text=t, rating=1) for i, t in enumerate(["لا يعمل", "؟؟؟", "!!!", "ok", "ممتاز"])]
    states = offline_workflow.run_batch(reviews)
    assert len(states) == 5 and all(s.agent3 is not None for s in states)
