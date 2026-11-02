"""Dedicated pages for the three agents. One workflow runs all three agents together (Agent 1 page);
the three pages show the SAME analysed review from each agent's point of view, so the agents stay
one connected system rather than three separate tools."""
from __future__ import annotations

from typing import Optional

import pandas as pd
import streamlit as st

from data_loader import row_to_review
from scenarios import get_scenarios
from schemas import ReviewInput, WorkflowState
from ui import components as C
from ui.pages import PAGES, ctx, remember, review_picker, store
from ui.result_view import _agent1_panel, _agent2_panel, _agent3_panel, run_live

EXAMPLES = {
    "Absher — renewal error, fees deducted": ("Absher", 2, "ابي اجدد الاقامة للعامل والتطبيق يقول حدث خطأ غير متوقع، حاولت اكثر من مره ونفس المشكله، والرسوم انخصمت من الحساب"),
    "Nafath — widespread login failure": ("Nafath", 1, "الكل عندهم نفس المشكله نفاذ ما يرسل الاشعار للجوال وما نقدر ندخل اي منصة حكومية"),
    "Sehhaty — booking not fulfilled + freezes": ("Sehhaty", 1, "حجزت موعد في المركز الصحي من التطبيق وطلع لي تم الحجز بس لما رحت قالوا ما فيه موعد باسمك، وفوق كذا التطبيق يعلق كل ما افتح صفحة المواعيد"),
    "Tawakkalna — praise only": ("Tawakkalna", 5, "التطبيق ممتاز وسهل الاستخدام الله يعطيكم العافية"),
}
PROTOTYPE = "Research Prototype — Proposed Decision-Support Solution"


def _link(page: str, label: str) -> None:
    if page in PAGES:
        st.page_link(PAGES[page], label=label, icon=":material/arrow_forward:")


def pick_state(key: str) -> Optional[WorkflowState]:
    """Choose which analysed review to inspect; the choice is shared by the three agent pages."""
    s = {k: v for k, v in store().items() if v.agent3 is not None}
    if not s:
        C.empty_state("🔎", "No analysed review yet", "Analyse a review on the Agent 1 page (or a sample on Results & Evaluation); "
                      "the three agent pages then show the same review from each agent's point of view.")
        _link("agent1", "Go to Agent 1 — Feedback Analysis")
        return None
    keys = list(s)
    cur = st.session_state.get("current_key")
    idx = keys.index(cur) if cur in keys else len(keys) - 1
    k = st.selectbox("Analysed review shown on the agent pages", keys, index=idx, key=f"pick_{key}",
                     format_func=lambda x: f"{s[x].review.review_id} · {s[x].review.app_name} · {s[x].review.text[:70]}")
    st.session_state["current_key"] = k
    return s[k]


def _header(agent: str, title: str, ar: str, subtitle: str) -> None:
    a = C.AGENTS[agent]
    C.hero(f"{a['icon']} {title}", subtitle, [PROTOTYPE, ar])


# =========================================================================== #
# Agent 1
# =========================================================================== #
def page_agent1() -> None:
    c = ctx()
    _header("Agent1", "Agent 1 — Feedback Analysis", "وكيل تحليل الملاحظات",
            "Reads the Arabic review, extracts each independent issue with verbatim evidence, classifies the feedback type and "
            "the service-quality (EGSQUAL) dimension, and flags ambiguity.")
    with st.container(border=True):
        st.markdown("**Arabic review input** · running it starts the full three-agent workflow")
        mode = st.segmented_control("Input", ["✍️ Type or load an example", "📚 Pick from dataset"],
                                    default="✍️ Type or load an example", key="an_mode")
        review: Optional[ReviewInput] = None
        ref_date = c.ref_date
        if mode == "📚 Pick from dataset":
            sub = review_picker("an")
            if sub is not None and not sub.empty:
                opts = sub.head(500)
                idx = st.selectbox("Review", opts.index, key="an_pick",
                                   format_func=lambda i: f"{opts.at[i, 'review_id']} · {opts.at[i, 'app_id']} · ★{opts.at[i, 'rating']} · {opts.at[i, 'text'][:80]}")
                review = row_to_review(opts.loc[idx])
        else:
            scen = {f"Test scenario {s.key} — {s.title}" + (" (synthetic)" if s.synthetic else ""): s for s in get_scenarios()}
            ex = st.selectbox("Load an example or a test scenario (S6 shows the review loop, S8 a simulated escalation)",
                              ["—"] + list(EXAMPLES) + list(scen), key="an_ex")
            if ex != "—" and st.session_state.get("an_ex_loaded") != ex:
                if ex in scen:
                    r = scen[ex].review
                    st.session_state.update({"an_text": r.text, "an_app": r.app_id, "an_rating": r.rating})
                else:
                    app, rating, text = EXAMPLES[ex]
                    st.session_state.update({"an_text": text, "an_app": app, "an_rating": rating})
                st.session_state["an_ex_loaded"] = ex
            txt = st.text_area("Arabic review", key="an_text", height=110, placeholder="اكتب مراجعة المستفيد هنا…")
            a, b = st.columns(2)
            st.session_state.setdefault("an_app", "CUSTOM")
            app = a.text_input("Application", key="an_app")
            rating = b.selectbox("Star rating", [None, 1, 2, 3, 4, 5], key="an_rating")
            sc = scen.get(ex)
            if sc is not None and txt.strip() == sc.review.text.strip():  # unchanged scenario: keep its id, flags and SLA date
                review, ref_date = sc.review, sc.reference_date or c.ref_date
                if sc.synthetic:
                    st.warning(f"🧪 Synthetic scenario: the review and its complaint record are artificial. SLA reference date fixed at {sc.reference_date}.")
            elif txt and txt.strip():
                n = st.session_state.get("typed_n", 0) + 1
                review = ReviewInput(review_id=f"TYPED-{n}", text=txt.strip(), app_id=app or "CUSTOM", app_name=app or "CUSTOM",
                                     rating=rating)
        if st.button("▶ Run the multi-agent workflow", type="primary", disabled=review is None, key="an_run"):
            if review.review_id.startswith("TYPED-"):
                st.session_state["typed_n"] = st.session_state.get("typed_n", 0) + 1
            state = run_live(c.workflow, review, ref_date, c.llm)
            remember([state], "single")
            st.session_state["current_key"] = f"single:{state.review.review_id}"

    state = pick_state("a1")
    if state is None:
        return
    C.fallback_banner(state, c.llm)
    C.workflow_diagram(state, llm=c.llm)
    with st.container(key="frame_Agent1_page"):
        _agent1_panel(state, c.llm)
    _link("agent2", "Next: Agent 2 — Impact & Prioritization")


# =========================================================================== #
# Agent 2
# =========================================================================== #
def page_agent2() -> None:
    c = ctx()
    _header("Agent2", "Agent 2 — Impact & Prioritization", "وكيل تقييم الأثر والأولوية",
            "Assesses each issue's impact on completing the service and its urgency, sets severity by rules R1–R4 and priority "
            "P1–P4 with a configurable Impact × Urgency matrix (informed by ITIL 4), and asks Agent 1 to re-check weak evidence.")
    state = pick_state("a2")
    if state is None:
        return
    C.fallback_banner(state, c.llm)
    C.section("Input from Agent 1", "issues, dimensions and evidence handed over")
    a1 = state.agent1
    if a1.issues:
        st.dataframe(pd.DataFrame([{"issue": i.issue_id, "type": i.issue_type, "dimension": i.dimension_name or i.dimension,
                                    "evidence (Arabic)": i.evidence, "verified": i.evidence_verified} for i in a1.issues]),
                     hide_index=True, width="stretch")
    else:
        st.caption(f"Agent 1 extracted no operational issue (feedback type: {a1.feedback_type}).")

    with st.container(key="frame_Agent2_page"):
        _agent2_panel(state, c.llm)

    if state.agent2.assessments:
        C.section("Impact × Urgency matrix", "research-defined; informed by ITIL 4 — not prescribed by it")
        matrix = state.run_config.get("matrix")
        for a in state.agent2.assessments:
            with st.container(border=True):
                g1, g2 = st.columns([2, 3])
                with g1:
                    st.markdown(f"**{a.issue_id}** · {a.issue_description}")
                    st.markdown(f"Impact **{a.impact or '—'}** × Urgency **{a.urgency or '—'}** → {C.priority_badge(a.priority.value)}",
                                unsafe_allow_html=True)
                    st.caption("Rules applied: " + ", ".join(a.rules_applied))
                    if "P-VERIFY" in a.rules_applied:
                        st.warning("Priority capped at P3 (P-VERIFY): the evidence stayed insufficient after the review loop.")
                with g2:
                    if matrix and a.impact:
                        C.matrix_grid(matrix, a.impact, a.urgency, "The outlined cell produced this issue's priority "
                                      "(matrix as configured when the review was analysed).")

    C.section("Requests for re-evaluation", "Agent 2 → Agent 1 when evidence is insufficient or contradictory")
    loop = [m for m in state.messages if m.message_type in ("REVIEW_REQUEST", "REVIEW_RESPONSE")]
    if loop:
        C.message_timeline(loop)
    else:
        st.success("No re-evaluation was requested: Agent 1's evidence was sufficient on the first pass.")
    _link("agent3", "Next: Agent 3 — Resolution & Escalation")


# =========================================================================== #
# Agent 3
# =========================================================================== #
def page_agent3() -> None:
    c = ctx()
    _header("Agent3", "Agent 3 — Resolution & Escalation", "وكيل المعالجة والتصعيد",
            "Decides whether the feedback is actionable, recommends the handling team, generates a simulated ticket with "
            "research-defined SLA targets, applies documented escalation rules and drafts an Arabic reply — all for human review.")
    state = pick_state("a3")
    if state is None:
        return
    C.fallback_banner(state, c.llm)
    st.info("🛡️ Human-in-the-loop: every ticket, escalation and reply below is a recommendation. A human officer reviews it "
            "before any real-world action; nothing is sent to AMER, the DGA or any government system.")
    C.section("Input from Agent 2", "severity, priority and impact per issue")
    a2 = state.agent2
    if a2.assessments:
        st.dataframe(pd.DataFrame([{"issue": a.issue_id, "severity": a.severity.value, "priority": a.priority.value,
                                    "impact": a.impact, "urgency": a.urgency, "service completion": a.service_completion,
                                    "verification": a.verification_status.value} for a in a2.assessments]),
                     hide_index=True, width="stretch")
    else:
        st.caption("Agent 2 assigned no severity (no operational issue).")
    with st.container(key="frame_Agent3_page"):
        _agent3_panel(state, c.llm)
