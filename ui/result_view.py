"""The multi-agent result screen: live run, workflow replay, the three agent panels,
communication, audit trail and export. Reads only the WorkflowState."""
from __future__ import annotations

import json
from datetime import date
from typing import Optional

import pandas as pd
import streamlit as st

from schemas import ReviewInput, WorkflowState
from ui import components as C
from workflow import state_to_dict, state_to_row


def run_live(workflow, review: ReviewInput, reference_date: Optional[date], llm) -> WorkflowState:
    """Run the real workflow while showing which agent is actually working (driven by workflow callbacks)."""
    box = st.empty()
    status = st.status("Agents are working…", expanded=False)

    def on_step(node: str, state: WorkflowState) -> None:
        with box.container():
            C.workflow_diagram(state, live_node=node, llm=llm)
        status.update(label=C.NODE_LABEL.get(node, node) + "…")
        status.write(f"Step {len(state.node_trace)}: {C.NODE_LABEL.get(node, node)}")

    state = workflow.run(review, reference_date=reference_date, on_step=on_step)
    box.empty()
    fell_back = any(e.action == "LLM_FALLBACK" for e in state.decision_log)
    status.update(label=f"Completed in {len(state.node_trace) - 1} step(s)" + (" — with LLM fallback" if fell_back else ""),
                  state="error" if (llm.available and fell_back) else "complete")
    return state


def render_result(state: WorkflowState, llm, key: str) -> None:
    a1, a2, a3, r = state.agent1, state.agent2, state.agent3, state.review
    if a1 is None or a2 is None or a3 is None:
        st.error("The workflow did not complete: " + "; ".join(state.errors))
        return
    C.fallback_banner(state, llm)

    # ---- Decision summary --------------------------------------------------
    sev = a2.overall_severity.value if a2.overall_severity else None
    pr = a2.overall_priority.value if a2.overall_priority else None
    prov = a2.verification_status.value == "INSUFFICIENT_EVIDENCE"
    c1, c2 = st.columns([3, 2])
    with c1:
        C.md(f'<div class="decision"><div class="small">FINAL OPERATIONAL DECISION (Agent 3)</div>'
             f'<div class="act">{C.esc(a3.action.value)}</div>'
             f'<div style="margin-top:8px">{C.severity_badge(sev, prov)} {C.priority_badge(pr)} '
             f'{C.pill("Escalation: " + a3.escalation.status.value, "warn" if a3.escalation.status.value != "NOT_ELIGIBLE" and a3.escalation.status.value != "NOT_APPLICABLE" else "muted")} '
             f'{C.pill("Human approval required", "navy") if a3.human_approval_required else ""}</div>'
             f'<div class="small" style="margin-top:8px">{C.esc(a3.action_rationale)}</div></div>')
    with c2:
        C.kpi_row([("Issues", len(a1.issues), a1.feedback_type), ("Revision rounds", f"{state.revision_count}/{state.max_revisions}",
                    "Agent 2 → Agent 1", "warn" if state.revision_count else ""), ("Tools called", len(state.tool_calls), "audited")])

    # ---- Workflow + step replay -------------------------------------------
    C.section("Multi-agent workflow", "driven by the real execution trace")
    steps = len(state.node_trace)
    replay = st.toggle("Step-by-step replay", key=f"replay_{key}",
                       help="Walk through the real execution trace one node at a time.")
    if replay and steps > 1:
        k = st.slider("Step", 1, steps, steps, key=f"step_{key}", format="Step %d") - 1
        C.workflow_diagram(state, step=k, llm=llm)
        node = state.node_trace[k]
        lo = state.node_log_index[k] if k < len(state.node_log_index) else 0
        hi = state.node_log_index[k + 1] if k + 1 < len(state.node_log_index) else len(state.decision_log)
        st.markdown(f"**Step {k + 1} · {C.NODE_LABEL.get(node, node)}** — {hi - lo} log entr{'y' if hi - lo == 1 else 'ies'}")
        entries = state.decision_log[lo:hi]
        if entries:
            st.dataframe(pd.DataFrame([{"agent": e.agent, "action": e.action, "detail": e.detail} for e in entries]),
                         hide_index=True, width="stretch", height=min(320, 38 + 35 * len(entries)))
    else:
        C.workflow_diagram(state, llm=llm)

    # ---- Agent panels -----------------------------------------------------
    t1, t2, t3, t4, t5, t6 = st.tabs(["🔵 Agent 1 · Investigation", "🟠 Agent 2 · Impact", "🟢 Agent 3 · Resolution",
                                       "💬 Communication", "🧾 Audit trail", "⬇️ Export"])
    with t1:
        _agent1_panel(state, llm)
    with t2:
        _agent2_panel(state, llm)
    with t3:
        _agent3_panel(state, llm)
    with t4:
        req = sum(1 for m in state.messages if m.message_type == "REVIEW_REQUEST")
        st.caption(f"{len(state.messages)} message(s) actually exchanged · {req} review request(s) · "
                   f"path: {' → '.join(state.node_trace)}")
        C.message_timeline(state.messages)
    with t5:
        agents = ["All", "Orchestrator", "Agent1", "Agent2", "Agent3"]
        who = st.segmented_control("Filter by agent", agents, default="All", key=f"who_{key}")
        log = pd.DataFrame([e.model_dump() for e in state.decision_log])
        tl = pd.DataFrame([t.model_dump() for t in state.tool_calls])
        if who and who != "All":
            log = log[log["agent"] == who]
            tl = tl[tl["agent"] == who] if not tl.empty else tl
        st.markdown(f"**Decision log** ({len(log)})")
        st.dataframe(log[["step", "agent", "action", "detail"]], hide_index=True, width="stretch", height=300)
        st.markdown(f"**Tool calls** ({len(tl)})")
        if not tl.empty:
            st.dataframe(tl, hide_index=True, width="stretch", height=260)
        if state.errors:
            st.markdown("**Errors / warnings**")
            for e in state.errors:
                st.error(e)
    with t6:
        d1, d2 = st.columns(2)
        d1.download_button("⬇️ Full result (JSON)", json.dumps(state_to_dict(state), ensure_ascii=False, indent=2),
                           file_name=f"result_{r.review_id}.json", mime="application/json", key=f"json_{key}", width="stretch")
        d2.download_button("⬇️ Summary row (CSV)", pd.DataFrame([state_to_row(state)]).to_csv(index=False).encode("utf-8-sig"),
                           file_name=f"result_{r.review_id}.csv", mime="text/csv", key=f"csv_{key}", width="stretch")


def _agent_header(state, agent: str, title: str, llm) -> None:
    label, kind = C.agent_reasoner(state, agent, llm)
    C.md(f'<div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:8px">'
         f'<h4 style="margin:4px 0">{C.esc(title)}</h4><div>Produced by: {C.pill(label, kind)}</div></div>')


def _agent1_panel(state, llm) -> None:
    a1, r = state.agent1, state.review
    _agent_header(state, "Agent1", "Feedback Investigation Agent", llm)
    meta = f"Review {r.review_id} · {r.app_name} ({r.app_id}) · rating {r.rating if r.rating is not None else 'n/a'}"
    st.caption(meta + (" · 🧪 SYNTHETIC review" if r.is_synthetic else ""))
    C.review_with_evidence(r.text, a1.issues)
    status_kind = {"ACCEPTED": "ok", "REVISED": "teal", "NEEDS_CLARIFICATION": "warn"}[a1.status.value]
    chips = (C.pill("Feedback type: " + a1.feedback_type, C.TYPE_KIND.get(a1.feedback_type, "outline"))
             + "".join(C.pill("also: " + t, "outline") for t in a1.secondary_types)
             + C.pill("Status: " + a1.status.value, status_kind)
             + (C.pill("Ambiguous / incomplete", "warn") if a1.is_ambiguous else C.pill("Specific", "ok")))
    C.md(f'<div style="margin:10px 0">{chips}</div>')
    if a1.ambiguity_reasons:
        st.warning("**Why clarification is needed:** " + " · ".join(a1.ambiguity_reasons))
    if a1.dialect_terms:
        C.md('<div class="small"><b>Saudi dialect recognised:</b> '
             + " ".join(C.pill(f"{k} → {v}", "outline") for k, v in a1.dialect_terms.items()) + "</div>")
    if not a1.issues:
        C.empty_state("🔍", "No operational issue extracted", "Agent 1 found no concrete, evidence-backed issue in this review.")
    for k, iss in enumerate(a1.issues):
        C.issue_card(iss, k)
    if len(state.agent1_history) > 1:
        with st.expander(f"Analysis versions across revision rounds ({len(state.agent1_history)})"):
            for h in state.agent1_history:
                st.markdown(f"- **Round {h.revision_round}** · type **{h.feedback_type}** · status {h.status.value} · "
                            f"issues {[i.issue_id for i in h.issues]}")


def _agent2_panel(state, llm) -> None:
    a2 = state.agent2
    _agent_header(state, "Agent2", "Impact Assessment & Prioritization Agent", llm)
    req = [m for m in state.messages if m.message_type == "REVIEW_REQUEST"]
    if req:
        codes = sorted({m.concern_code for m in req})
        st.error(f"**Agent 2 requested a revision from Agent 1** — {len(req)} concern(s) over {state.revision_count} round(s): "
                 f"{', '.join(codes)}. See the Communication tab for the full exchange.", icon=":material/sync_problem:")
    else:
        st.success("**No revision requested** — Agent 2 found Agent 1's evidence sufficient on the first pass.", icon=":material/check_circle:")

    c1, c2 = st.columns(2)
    sev = a2.overall_severity.value if a2.overall_severity else None
    with c1:
        C.md(f'<div class="card"><h4>Operational severity</h4><div class="small">Decided by impact rules R1–R4 (never by tone or rating)</div>'
             f'<div style="margin-top:10px">{C.severity_badge(sev, a2.verification_status.value == "INSUFFICIENT_EVIDENCE")} '
             f'{C.priority_badge(a2.overall_priority.value if a2.overall_priority else None)}</div>{C.severity_meter(sev)}'
             f'<div class="small" style="margin-top:8px">Verification: {C.pill(a2.verification_status.value, C.VS_KIND[a2.verification_status.value])}</div></div>')
    with c2:
        score = a2.sentiment_score
        pos = int((score + 1) / 2 * 100)
        C.md(f'<div class="card"><h4>Sentiment (reported separately)</h4><div class="small">Tone of the text and star rating — informational only (rule S0)</div>'
             f'<div style="margin-top:10px">{C.pill(a2.sentiment_intensity, "navy")} <span class="small">score {score:+.2f}</span></div>'
             f'<div style="position:relative;height:10px;border-radius:5px;margin-top:10px;background:linear-gradient(90deg,#D03B3B,#EDF3EF,#0CA30C)">'
             f'<div style="position:absolute;left:calc({pos}% - 6px);top:-3px;width:12px;height:16px;border-radius:3px;background:#0E3B2C"></div></div>'
             f'<div style="display:flex;justify-content:space-between" class="small"><span>negative</span><span>positive</span></div></div>')
    if a2.sentiment_vs_severity_note:
        st.info(a2.sentiment_vs_severity_note, icon="⚖️")
    if not a2.assessments:
        C.empty_state("📭", "Nothing to assess", "No operational issue was extracted, so no severity or priority was assigned.")
    issues = {i.issue_id: i for i in state.agent1.issues}
    for a in a2.assessments:
        C.assessment_card(a, issues.get(a.issue_id))


def _agent3_panel(state, llm) -> None:
    a3 = state.agent3
    _agent_header(state, "Agent3", "Resolution & Escalation Agent", llm)
    st.warning("All tickets and escalations are **SIMULATED**. Nothing is submitted to AMER, the DGA or any government system. "
               "Escalation rules are proposed research rules, not official DGA rules.", icon="🛡️")
    c1, c2 = st.columns([3, 2])
    with c1:
        sec = "".join(f"<li>{C.esc(s)}</li>" for s in a3.secondary_actions) or "<li>None</li>"
        C.md(f'<div class="decision"><div class="small">RECOMMENDED ACTION</div><div class="act">{C.esc(a3.action.value)}</div>'
             f'<div style="margin-top:6px">{C.pill("Actionable" if a3.actionable else "Not actionable yet", "ok" if a3.actionable else "warn")}'
             f'{C.pill("Human approval required", "navy") if a3.human_approval_required else ""}</div>'
             f'<div style="margin-top:10px"><b>Justification.</b> {C.esc(a3.action_rationale)}</div>'
             f'<div style="margin-top:10px"><b>Secondary actions</b><ul style="margin:4px 0 0 0">{sec}</ul></div></div>')
    with c2:
        e = a3.escalation
        kind = {"CANDIDATE_AWAITING_HUMAN_APPROVAL": "bad", "PENDING_VERIFICATION": "warn"}.get(e.status.value, "muted")
        extra = ""
        if e.complaint_reference:
            extra += f'<div class="small">Complaint reference: <b>{C.esc(e.complaint_reference)}</b></div>'
        if e.sla_deadline:
            extra += f'<div class="small">SLA deadline: <b>{C.esc(e.sla_deadline)}</b> · overdue {e.sla_days_overdue} day(s)</div>'
        if e.missing_fields:
            extra += '<div class="small" style="margin-top:4px">Missing (not invented): ' + "".join(C.pill(f, "warn") for f in e.missing_fields) + "</div>"
        C.md(f'<div class="card"><h4>Escalation eligibility</h4>{C.pill(e.status.value, kind)}'
             f'<div class="small" style="margin-top:6px">Channel: AMER / Balaagh Digital — <b>simulated only</b></div>{extra}'
             f'<div class="small" style="margin-top:6px">Human approval before any external escalation: <b>mandatory</b></div></div>')
    if e.rules:
        st.markdown("**Escalation rule check (E1–E5)**")
        C.rules_checklist(e)
    if e.complaint_record:
        with st.expander("Complaint history record (SYNTHETIC data store)"):
            st.json(e.complaint_record.model_dump())
    if a3.ticket:
        st.markdown("**Simulated service ticket**")
        C.ticket_card(a3.ticket)
    else:
        C.empty_state("✅", "No ticket", "No action is required for this feedback, so no ticket was generated.")
