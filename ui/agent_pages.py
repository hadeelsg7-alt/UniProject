"""The three agent pages. Every page has the same structure: title + one line, 3-4 indicators, one concise
results table over all analysed reviews, a Review ID selector, and a collapsed "View Details" section.
All three show results of ONE shared workflow run (Data & Run Analysis), so the agents stay connected."""
from __future__ import annotations


import pandas as pd
import streamlit as st

from schemas import EscalationStatus, WorkflowState
from ui import components as C
from ui.pages import _goto, _rtl_table, analysed, ctx, presentation
from ui.result_view import _agent1_panel, _agent2_panel, _agent3_panel

ESC_TEXT = {
    EscalationStatus.CANDIDATE_AWAITING_HUMAN_APPROVAL: "Recommend escalation (awaiting human approval)",
    EscalationStatus.PENDING_VERIFICATION: "Verify the previous complaint first",
    EscalationStatus.NOT_ELIGIBLE: "Not eligible",
    EscalationStatus.NOT_APPLICABLE: "Not applicable",
}


def _no_results() -> None:
    C.empty_state("🔎", "No analysed reviews yet", "Run the analysis on the Data & Run Analysis page; this page then shows "
                  "this agent's part of the shared workflow.")
    _goto("data", "Go to Data & Run Analysis")


def _select(states: list[WorkflowState], key: str) -> WorkflowState:
    """Review ID selector; the choice is shared by the three agent pages."""
    ids = {s.review.review_id: s for s in states}
    cur = st.session_state.get("current_rid")
    keys = list(ids)
    idx = keys.index(cur) if cur in keys else len(keys) - 1
    rid = st.selectbox("Review ID", keys, index=idx, key=f"rid_{key}",
                       format_func=lambda k: f"{k} · {ids[k].review.app_name} · {ids[k].review.text[:60]}")
    st.session_state["current_rid"] = rid
    return ids[rid]


def _review_box(state: WorkflowState) -> None:
    r = state.review
    st.caption(f"{r.review_id} · {r.app_name} · rating {r.rating if r.rating is not None else 'n/a'}"
               + (" · 🧪 synthetic" if r.is_synthetic else ""))
    C.review_with_evidence(r.text, state.agent1.issues)


def _trace(state: WorkflowState, agents: tuple[str, ...]) -> None:
    if presentation():
        return
    log = pd.DataFrame([e.model_dump() for e in state.decision_log if e.agent in agents])
    if not log.empty:
        st.markdown("**Decision log**")
        st.dataframe(log.assign(agent=log["agent"].map(C.agent_name))[["step", "agent", "action", "detail"]],
                     hide_index=True, width="stretch", height=220)


# =========================================================================== #
# Agent 1
# =========================================================================== #
def page_agent1() -> None:
    c = ctx()
    C.page_header("🔍 Agent 1 — Feedback Analysis · وكيل تحليل الملاحظات",
                  "Understands the Arabic review, extracts independent issues with verbatim evidence, classifies the feedback type "
                  "and the EGSQUAL service-quality dimension, and flags ambiguous or incomplete feedback.", C.AGENTS["Agent1"]["color"])
    states = analysed()
    if not states:
        return _no_results()
    issues = sum(len(s.agent1.issues) for s in states)
    C.kpi_row([("Reviews analysed", len(states), "input to Agent 1"),
               ("Issues extracted", issues, f"{issues / len(states):.2f} per review"),
               ("Multi-issue reviews", sum(len(s.agent1.issues) >= 2 for s in states), "2+ independent issues"),
               ("Ambiguous / incomplete", sum(s.agent1.is_ambiguous for s in states), "need clarification", "warn")])
    st.dataframe(pd.DataFrame([{"Review ID": s.review.review_id, "Application": s.review.app_name,
                                "Feedback type": s.agent1.feedback_type, "Issues": len(s.agent1.issues),
                                "EGSQUAL dimensions": ", ".join(dict.fromkeys(i.dimension_name.split(" (")[0] for i in s.agent1.issues)) or "—",
                                "Ambiguous": "yes" if s.agent1.is_ambiguous else "no", "Status": s.agent1.status.value}
                               for s in states]), hide_index=True, width="stretch", height=min(320, 38 + 35 * len(states)))
    state = _select(states, "a1")
    _review_box(state)
    if state.agent1.issues:
        _rtl_table(pd.DataFrame([{"Issue": i.issue_id, "Type": i.issue_type, "Dimension": i.dimension_name.split(" (")[0],
                                  "EGSQUAL (provisional)": i.egsqual_candidate or "—", "Evidence": i.evidence}
                                 for i in state.agent1.issues]), ("Evidence",))
    else:
        st.caption(f"No operational issue extracted · feedback type {state.agent1.feedback_type}"
                   + (" · ambiguous: " + "; ".join(state.agent1.ambiguity_reasons) if state.agent1.is_ambiguous else ""))
    with st.expander("View Details — reasoning, dimension definitions, dialect terms, trace"):
        _agent1_panel(state, c.llm)
        _trace(state, ("Agent1",))


# =========================================================================== #
# Agent 2
# =========================================================================== #
def page_agent2() -> None:
    c = ctx()
    C.page_header("⚖️ Agent 2 — Impact & Prioritization · وكيل تقييم الأثر والأولوية",
                  "Assesses impact and urgency from Agent 1's evidence, sets severity (rules R1–R4) and priority P1–P4 with a "
                  "research-defined Impact × Urgency matrix (informed by ITIL 4 and ISO/IEC 20000-1), and asks Agent 1 to re-check "
                  "weak evidence.", C.AGENTS["Agent2"]["color"])
    states = analysed()
    if not states:
        return _no_results()
    rows = [(s, a) for s in states for a in s.agent2.assessments]
    C.kpi_row([("Issues assessed", len(rows), f"from {len(states)} review(s)"),
               ("High / Critical", sum(a.severity.value in ("High", "Critical") for _, a in rows), "severity"),
               ("P1 / P2", sum(a.priority.value in ("P1", "P2") for _, a in rows), "priority", "warn"),
               ("Low evidence confidence", sum(a.evidence_confidence == "Low" for _, a in rows), "provisional · human check", "crit")])
    st.caption(f"Agent 2 sent {sum(m.message_type == 'REVIEW_REQUEST' for s in states for m in s.messages)} re-evaluation request(s) "
               "to Agent 1 (max 2 rounds per review). Missing information never lowers a priority.")
    if rows:
        st.dataframe(pd.DataFrame([{"Review ID": s.review.review_id, "Issue": a.issue_id, "Impact": a.impact, "Urgency": a.urgency,
                                    "Severity": a.severity.value + (" (provisional)" if a.provisional else ""),
                                    "Priority": a.priority.value, "Evidence confidence": a.evidence_confidence,
                                    "Decision": f"{a.service_completion} · {a.scope}" + (f" · {', '.join(a.risk_flags)}" if a.risk_flags else "")}
                                   for s, a in rows]), hide_index=True, width="stretch", height=min(320, 38 + 35 * len(rows)))
    state = _select(states, "a2")
    _review_box(state)
    a2 = state.agent2
    if a2.assessments:
        _rtl_table(pd.DataFrame([{"Issue": a.issue_id, "Impact × Urgency": f"{a.impact} × {a.urgency}", "Severity":
                                  a.severity.value + (" (provisional)" if a.provisional else ""), "Priority": a.priority.value,
                                  "Confidence": a.evidence_confidence, "Evidence": a.evidence[0] if a.evidence else ""}
                                 for a in a2.assessments]), ("Evidence",))
    else:
        st.caption(f"No severity assigned (no operational issue). Sentiment reported separately: {a2.sentiment_intensity}.")
    with st.expander("View Details — matrix, applied rules, impact reasoning, collaboration trace"):
        matrix = state.run_config.get("matrix")
        for a in a2.assessments:
            st.markdown(f"**{a.issue_id}** · rules applied: {', '.join(a.rules_applied)}")
            if matrix and a.impact:
                C.matrix_grid(matrix, a.impact, a.urgency, "Outlined cell = this issue's priority (matrix as configured at analysis time).")
        _agent2_panel(state, c.llm)
        loop = [m for m in state.messages if m.message_type in ("REVIEW_REQUEST", "REVIEW_RESPONSE")]
        st.markdown("**Agent 1 ↔ Agent 2 collaboration**")
        if loop:
            C.message_timeline(loop)
        else:
            st.caption("No re-evaluation was needed for this review.")
        _trace(state, ("Agent2",))


# =========================================================================== #
# Agent 3
# =========================================================================== #
def _sla_text(t) -> str:
    return f"respond {t.sla_response_hours} h · resolve {t.sla_resolution_hours} h" if t is not None and t.sla_response_hours else "—"


def page_agent3() -> None:
    c = ctx()
    C.page_header("🎫 Agent 3 — Resolution & Escalation · وكيل المعالجة والتصعيد",
                  "Decides whether feedback is actionable, recommends the handling team and action, creates a simulated ticket with "
                  "research-defined SLA targets, evaluates escalation and drafts an Arabic reply — for human review (informed by "
                  "ISO 10002 and ITIL 4).", C.AGENTS["Agent3"]["color"])
    states = analysed()
    if not states:
        return _no_results()
    a3s = [s.agent3 for s in states]
    C.kpi_row([("Actionable feedback", sum(a.actionable for a in a3s), f"of {len(states)} review(s)"),
               ("Simulated tickets", sum(a.ticket is not None for a in a3s), "awaiting human review"),
               ("Escalation recommended", sum(a.escalation.status == EscalationStatus.CANDIDATE_AWAITING_HUMAN_APPROVAL for a in a3s),
                "simulated · human approval", "warn"),
               ("Verify complaint first", sum(a.escalation.status == EscalationStatus.PENDING_VERIFICATION for a in a3s),
                "details missing, nothing invented")])
    st.dataframe(pd.DataFrame([{"Review ID": s.review.review_id, "Recommended action": s.agent3.action.value,
                                "Handling team": s.agent3.handling_team or "—",
                                "Ticket status": "Simulated · awaiting human review" if s.agent3.ticket else "No ticket",
                                "Priority": s.agent2.overall_priority.value if s.agent2.overall_priority else "—",
                                "SLA target (simulated)": _sla_text(s.agent3.ticket),
                                "Escalation": ESC_TEXT[s.agent3.escalation.status]} for s in states]),
                 hide_index=True, width="stretch", height=min(320, 38 + 35 * len(states)))
    state = _select(states, "a3")
    a3 = state.agent3
    _rtl_table(pd.DataFrame([{"Recommended action": a3.action.value, "Handling team": a3.handling_team or "—",
                              "Priority": state.agent2.overall_priority.value if state.agent2.overall_priority else "—",
                              "SLA target (simulated)": _sla_text(a3.ticket), "Escalation": ESC_TEXT[a3.escalation.status]}]), ())
    if a3.response_draft_ar:
        C.arabic_draft(a3.response_draft_ar)
    with st.expander("View Details — ticket, escalation rules E1–E5, rationale, trace"):
        _agent3_panel(state, c.llm)
        _trace(state, ("Agent3",))
    st.caption("🛡️ Every ticket, escalation and reply is simulated and requires human review; nothing is sent to AMER, the DGA "
               "or any government system.")
