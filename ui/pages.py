"""Dashboard pages. Shared context (LLM client, workflow, dataset) is put in
st.session_state["ctx"] by app.py. Results of every run are kept in a session
store so the overview dashboard reflects only analyses that really ran."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Optional

import pandas as pd
import streamlit as st

import tools
from data_loader import row_to_review
from scenarios import get_scenarios
from schemas import ReviewInput, WorkflowState
from ui import charts as CH
from ui import components as C
from ui import theme as T
from ui.result_view import render_result, run_live
from workflow import state_to_row


@dataclass
class Ctx:
    llm: object
    workflow: object
    df: Optional[pd.DataFrame]
    info: Optional[dict]
    ref_date: date


def ctx() -> Ctx:
    return st.session_state["ctx"]


# --------------------------------------------------------------------------- #
# Session result store
# --------------------------------------------------------------------------- #
def store() -> dict[str, WorkflowState]:
    return st.session_state.setdefault("store", {})


def remember(states: list[WorkflowState], source: str) -> None:
    s = store()
    for st_ in states:
        if st_.agent3 is not None:
            s[f"{source}:{st_.review.review_id}"] = st_


def issue_frame(states: list[WorkflowState]) -> pd.DataFrame:
    rows = []
    for s in states:
        if not s.agent1 or not s.agent2:
            continue
        issues = {i.issue_id: i for i in s.agent1.issues}
        for a in s.agent2.assessments:
            i = issues.get(a.issue_id)
            rows.append({"review_id": s.review.review_id, "issue_type": i.issue_type if i else None,
                         "dimension": (i.dimension_name or i.dimension) if i else None,
                         "severity": a.severity.value, "priority": a.priority.value})
    return pd.DataFrame(rows, columns=["review_id", "issue_type", "dimension", "severity", "priority"])


def run_batch_with_progress(reviews: list[ReviewInput], label: str) -> list[WorkflowState]:
    bar = st.progress(0.0, text=f"{label}: 0/{len(reviews)}")
    states = ctx().workflow.run_batch(reviews, reference_date=ctx().ref_date,
                                      progress=lambda k, n: bar.progress(k / n, text=f"{label}: {k}/{n}"))
    bar.empty()
    fb = sum(1 for s in states if any(e.action == "LLM_FALLBACK" for e in s.decision_log))
    if ctx().llm.available and fb:
        st.error(f"⛔ {fb} of {len(states)} review(s) had at least one LLM failure and used the rule-based fallback for that step. "
                 "Open a result to see which step.")
    return states


def review_picker(prefix: str, limit: int = 500) -> Optional[pd.DataFrame]:
    df = ctx().df
    if df is None:
        C.empty_state("📁", "No dataset loaded", "Upload the reviews CSV (sidebar or the Upload tab) or place it at data/reviews.csv.")
        return None
    f1, f2, f3 = st.columns([2, 1, 2])
    apps = f1.multiselect("Application", sorted(df["app_id"].unique()), key=f"{prefix}_apps")
    ratings = f2.multiselect("Star rating", sorted(df["rating"].dropna().unique().tolist()), key=f"{prefix}_ratings")
    search = f3.text_input("Search Arabic text", key=f"{prefix}_search", placeholder="مثال: رمز التحقق")
    sub = df
    if apps:
        sub = sub[sub["app_id"].isin(apps)]
    if ratings:
        sub = sub[sub["rating"].isin(ratings)]
    if search:
        sub = sub[sub["text"].str.contains(search, regex=False, na=False)]
    st.caption(f"{len(sub):,} matching review(s)")
    return sub


def agent_results(states: list[WorkflowState], key: str, show_rating_chart: bool = True) -> None:
    """KPIs and charts grouped by the agent that produced them, each in that agent's coloured frame."""
    rows = pd.DataFrame([state_to_row(s) for s in states])
    iss = issue_frame(states)
    n = max(len(rows), 1)
    A1, A2, A3 = (C.AGENTS[k]["color"] for k in ("Agent1", "Agent2", "Agent3"))

    C.kpi_row([("Reviews analysed", len(rows), f"{rows['review_id'].nunique()} unique · input to all three agents", "navy")])
    st.write("")

    # ---- Agent 1: Investigation -------------------------------------------
    with st.container(key=f"frame_Agent1_res_{key}"):
        C.agent_section_header("Agent1", "issues, feedback types, EGSQUAL dimensions, ambiguity")
        C.kpi_row([("Issues extracted", len(iss), f"{len(iss) / n:.2f} per review"),
                   ("Multi-issue reviews", int((rows["n_issues"] >= 2).sum()), "2+ independent issues"),
                   ("Need clarification", int((rows["analysis_status"] == "NEEDS_CLARIFICATION").sum()), "ambiguous / incomplete")])
        st.write("")
        g1, g2 = st.columns(2)
        with g1, st.container(border=True):
            st.markdown("**Feedback categories** · primary type per review")
            st.altair_chart(CH.single_hue_bars(rows["feedback_type"], A1, "Reviews"), width="stretch")
        with g2, st.container(border=True):
            st.markdown("**Service-quality dimensions (EGSQUAL)** · per extracted issue")
            if iss.empty:
                st.caption("No issues extracted yet.")
            else:
                st.altair_chart(CH.single_hue_bars(iss["dimension"], A1, "Issues"), width="stretch")

    # ---- Agent 2: Impact --------------------------------------------------
    with st.container(key=f"frame_Agent2_res_{key}"):
        C.agent_section_header("Agent2", "severity, priority, sentiment vs severity, review requests")
        hi = int(iss["severity"].isin(["High", "Critical"]).sum())
        urgent = int(iss["priority"].isin(["P1", "P2"]).sum())
        C.kpi_row([("High / Critical issues", hi, f"{hi / max(len(iss), 1) * 100:.0f}% of issues"),
                   ("P1 / P2 issues", urgent, "urgent priorities"),
                   ("Revision requests", int((rows["revision_rounds"] > 0).sum()), "reviews sent back to the Investigation Agent")])
        st.write("")
        g3, g4 = st.columns(2)
        with g3, st.container(border=True):
            st.markdown("**Issue severity** · operational impact (rules R1–R4)")
            st.altair_chart(CH.severity_bars(iss["severity"]), width="stretch")
        with g4, st.container(border=True):
            st.markdown("**Issue priority** · P1 (most urgent) → P4")
            st.altair_chart(CH.priority_bars(iss["priority"]), width="stretch")
        if show_rating_chart:
            with st.container(border=True):
                st.markdown("**Severity is not the star rating** · share of each overall severity within each rating")
                st.altair_chart(CH.severity_by_rating(rows), width="stretch")
                st.caption("'No severity' = no concrete operational issue was found (e.g. praise or vague complaints).")

    # ---- Agent 3: Resolution ----------------------------------------------
    with st.container(key=f"frame_Agent3_res_{key}"):
        C.agent_section_header("Agent3", "recommended actions, escalation eligibility, simulated tickets")
        tickets = sum(1 for s in states if s.agent3 and s.agent3.ticket)
        esc_n = int(rows["escalation_status"].isin(["CANDIDATE_AWAITING_HUMAN_APPROVAL", "PENDING_VERIFICATION"]).sum())
        C.kpi_row([("Simulated tickets", tickets, "none submitted externally"),
                   ("Escalation candidate / pending", esc_n, "always needs human approval"),
                   ("No action required", int((rows["action"] == "No action required").sum()), "e.g. praise only")])
        st.write("")
        g5, g6 = st.columns(2)
        with g5, st.container(border=True):
            st.markdown("**Recommended action** · per review")
            st.altair_chart(CH.single_hue_bars(rows["action"], A3, "Reviews"), width="stretch")
        with g6, st.container(border=True):
            st.markdown("**Escalation status** · per review (simulated)")
            st.altair_chart(CH.single_hue_bars(rows["escalation_status"], A3, "Reviews"), width="stretch")
    return rows


# =========================================================================== #
# 1. Overview dashboard
# =========================================================================== #
def page_dashboard() -> None:
    c = ctx()
    C.hero("Arabic Beneficiary Feedback — Service-Quality Intelligence",
           "Multi-agent analysis of Saudi government mobile-app reviews: issue investigation, impact prioritisation and simulated resolution.",
           ["3 cooperating agents", "EGSQUAL dimensions", "Evidence-grounded", "Human-in-the-loop", "Simulated escalation only"])
    C.mode_banner(c.llm)
    C.section("The three agents", "who does what")
    C.agents_overview()

    C.section("Dataset", "the uploaded corpus (no ground-truth labels)")
    if c.info:
        df = c.df
        C.kpi_row([("Reviews in file", f"{c.info['rows_in_file']:,}", "original rows", "navy"),
                   ("Unique reviews", f"{c.info['rows_available']:,}", f"{c.info['duplicates_dropped']} duplicates removed", "navy"),
                   ("Applications", c.info["apps"], "government apps", "navy"),
                   ("Average rating", f"{df['rating'].mean():.2f} ★", f"{(df['rating'] <= 2).mean() * 100:.0f}% rated 1–2 ★", "navy")])
        with st.expander("Rating and application distribution", expanded=False):
            a, b = st.columns(2)
            with a:
                st.altair_chart(CH.single_hue_bars(df["rating"].astype("Int64").astype(str) + "★", T.NAVY, "Reviews",
                                                   order=["1★", "2★", "3★", "4★", "5★"]), width="stretch")
            with b:
                top = df["app_id"].value_counts().head(10)
                st.altair_chart(CH.single_hue_bars(df[df["app_id"].isin(top.index)]["app_id"], T.NAVY, "Reviews (top 10 apps)"),
                                width="stretch")
    else:
        C.empty_state("📁", "No dataset loaded", "Upload the CSV from the sidebar to see dataset statistics.")

    C.section("Analysis results", "computed only from reviews actually analysed in this session")
    states = list(store().values())
    with st.container(border=True):
        a, b, d = st.columns([2, 1, 1])
        default_n = 200 if not c.llm.available else 10
        n = a.number_input("Analyse a random sample of reviews from the dataset", 1, 2000, default_n, step=10,
                           help="LLM mode makes about 2–4 API calls per review." if c.llm.available else "Offline mode: ~20 ms per review.")
        b.write("")
        b.write("")
        go = b.button("▶ Run sample analysis", type="primary", disabled=c.df is None, width="stretch")
        d.write("")
        d.write("")
        if d.button("Clear results", disabled=not states, width="stretch"):
            st.session_state["store"] = {}
            st.rerun()
        if go:
            sample = c.df.sample(min(int(n), len(c.df)), random_state=None)
            remember(run_batch_with_progress([row_to_review(r) for _, r in sample.iterrows()], "Analysing"), "dashboard")
            st.rerun()

    states = list(store().values())
    if not states:
        C.empty_state("📊", "No analysis yet", "Run a sample above or analyse a single review — the KPIs and charts "
                      "are computed from those real results only.")
        return
    agent_results(states, "dash")
    st.download_button("⬇️ Download all results (CSV)", pd.DataFrame([state_to_row(x) for x in states]).to_csv(index=False).encode("utf-8-sig"),
                       "analysis_results.csv", "text/csv")


# =========================================================================== #
# 2. Analyse feedback
# =========================================================================== #
EXAMPLES = {
    "Absher — renewal error, fees deducted": ("Absher", 2, "ابي اجدد الاقامة للعامل والتطبيق يقول حدث خطأ غير متوقع، حاولت اكثر من مره ونفس المشكله، والرسوم انخصمت من الحساب"),
    "Nafath — widespread login failure": ("Nafath", 1, "الكل عندهم نفس المشكله نفاذ ما يرسل الاشعار للجوال وما نقدر ندخل اي منصة حكومية"),
    "Sehhaty — booking not fulfilled + freezes": ("Sehhaty", 1, "حجزت موعد في المركز الصحي من التطبيق وطلع لي تم الحجز بس لما رحت قالوا ما فيه موعد باسمك، وفوق كذا التطبيق يعلق كل ما افتح صفحة المواعيد"),
    "Tawakkalna — praise only": ("Tawakkalna", 5, "التطبيق ممتاز وسهل الاستخدام الله يعطيكم العافية"),
}


def _scenario_label(s) -> str:
    return f"Test scenario {s.key} — {s.title}" + (" (synthetic)" if s.synthetic else "")


def page_analyze() -> None:
    c = ctx()
    C.hero("Analyse Beneficiary Feedback", "Type an Arabic review, load an example or test scenario, or pick one from the dataset — then watch the three agents work.",
           ["RTL Arabic", "Saudi dialect", "Live agent trace"])
    C.mode_banner(c.llm)
    C.agents_overview(compact=True)
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
        scen = {_scenario_label(s): s for s in get_scenarios()}
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
            review = ReviewInput(review_id="TYPED-1", text=txt.strip(), app_id=app or "CUSTOM", app_name=app or "CUSTOM", rating=rating)

    if review is not None:
        C.section("Original review", "right-to-left")
        C.md(f'<div class="review-box"><div class="rtl">{C.esc(review.text)}</div></div>')
    run = st.button("▶ Run Multi-Agent Analysis", type="primary", disabled=review is None, key="an_run")
    if run and review is not None:
        state = run_live(c.workflow, review, ref_date, c.llm)
        st.session_state["an_state"] = state
        remember([state], "single")
    if "an_state" in st.session_state:
        render_result(st.session_state["an_state"], c.llm, "an")


# =========================================================================== #
# 3. Methodology & rules
# =========================================================================== #
def page_methodology() -> None:
    C.hero("Methodology & Rules", "How the framework decides — every rule below is editable in knowledge_base/ without code changes.",
           ["Proposed research rules", "Not official DGA rules"])
    C.md('<div class="card"><h4>Architecture</h4><div class="small">'
         '<b>Agent 1</b> extracts independent issues with verbatim evidence, maps them to EGSQUAL via the knowledge-base tool, '
         'self-verifies and flags ambiguity. <b>Agent 2</b> judges operational impact; severity and priority come from rules R1–R4 '
         'and P-rules, never from sentiment. It verifies evidence and can send review requests back to Agent 1 (max 2 rounds). '
         '<b>Agent 3</b> checks complaint history and SLA (when available), applies escalation rules E1–E5 and produces a '
         '<b>simulated</b> ticket that always needs human approval. A shared structured state records every message, tool call and decision.'
         '</div></div>')
    kb = tools.egsqual_kb()
    C.section("EGSQUAL service-quality dimensions")
    st.dataframe(pd.DataFrame([{"key": k, "dimension": v["name_en"], "arabic": v["name_ar"], "definition": v["definition"], "source": v["source"]}
                               for k, v in kb.items()]), hide_index=True, width="stretch")
    sr = tools.severity_rules_kb()
    C.section("Severity & priority rules (Agent 2)")
    st.info(sr["principle"])
    a, b = st.columns(2)
    a.dataframe(pd.DataFrame(sr["severity_rules"]), hide_index=True, width="stretch")
    b.dataframe(pd.DataFrame(sr["priority_rules"]), hide_index=True, width="stretch")
    er = tools.escalation_rules_kb()
    C.section("Escalation eligibility rules (Agent 3)")
    st.warning("Proposed, configurable research rules — not official rules of the DGA or the AMER / Balaagh Digital service. "
               "Escalation is simulated only.")
    st.dataframe(pd.DataFrame(er["rules"]), hide_index=True, width="stretch")
    st.caption(f"Minimum severity {er['min_severity_for_escalation']} · default SLA {er['default_response_sla_days']} days "
               f"({er['default_sla_note']}) · human approval required: {er['require_human_approval']}")
