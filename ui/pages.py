"""Dashboard pages. Shared context (LLM client, workflow, dataset) is put in
st.session_state["ctx"] by app.py. Results of every run are kept in a session
store so the overview dashboard reflects only analyses that really ran."""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date
from typing import Optional

import pandas as pd
import streamlit as st

import config
import tools
from data_loader import row_to_review
from schemas import ReviewInput, WorkflowState
from ui import charts as CH
from ui import components as C
from ui import theme as T

from workflow import state_to_dict, state_to_row


@dataclass
class Ctx:
    llm: object
    workflow: object
    df: Optional[pd.DataFrame]
    info: Optional[dict]
    ref_date: date
    matrix: Optional[dict] = None       # Impact x Urgency cells in use
    sla_targets: Optional[dict] = None  # simulated SLA targets in use


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
# 1. Executive Dashboard (homepage)
# =========================================================================== #
PAGES: dict = {}  # name -> st.Page, filled by app.py so pages can link to each other
PROTOTYPE = "Research Prototype — Proposed Decision-Support Solution"


def _goto(page: str, label: str) -> None:
    if page in PAGES:
        st.page_link(PAGES[page], label=label, icon=":material/arrow_forward:")


def page_executive() -> None:
    import insights as INS

    c = ctx()
    C.hero("GovFeedback AI", "Multi-Agent Intelligence for Digital Service Quality — what beneficiaries report about Saudi "
           "government mobile services, and where to act first.", [PROTOTYPE])

    C.section("Full dataset", "all reviews in the loaded file · before any analysis")
    if c.info:
        df = c.df
        C.kpi_row([("Total reviews", f"{c.info['rows_in_file']:,}", "rows in the file", "navy"),
                   ("Unique reviews", f"{c.info['rows_available']:,}", f"{c.info['duplicates_dropped']} duplicates removed", "navy"),
                   ("Applications", c.info["apps"], "government apps", "navy"),
                   ("Average rating", f"{df['rating'].mean():.2f} ★", f"{(df['rating'] <= 2).mean() * 100:.0f}% rated 1–2 ★", "navy")])
    else:
        C.empty_state("📁", "No dataset loaded", "Upload the reviews CSV in Settings (bottom of the sidebar).")

    states = [s for s in store().values() if s.agent3 is not None]
    total = c.info["rows_available"] if c.info else None
    C.section("Analysed feedback", f"{len(states):,} review(s) analysed by the three agents"
              + (f" · {len(states) / total * 100:.1f}% of the unique reviews" if total else ""))
    if not states:
        C.empty_state("📊", "No analysis yet — no insights are shown", "Insights appear only after reviews have been analysed. "
                      "Analyse a single review on the Agent 1 page, or a sample on Results & Evaluation.")
        _goto("agent1", "Analyse a review — Agent 1")
        _goto("results", "Analyse a sample — Results & Evaluation")
        return
    ins = INS.compute_insights(states)
    cov = ins["coverage"]
    urgent = sum(n for p, n, _ in ins["priority"] if p in INS.URGENT)
    C.kpi_row([("Reviews analysed", cov["reviews"], f"{cov['apps']} application(s)"),
               ("Service issues", cov["issues"], "evidence-backed"),
               ("Actionable feedback", ins["actionable"], f"{ins['actionable'] / cov['reviews'] * 100:.0f}% of analysed reviews"),
               ("P1 / P2 issues", urgent, "require attention first", "warn" if urgent else "")])
    st.write("")
    for line in ins["summary"][:3]:
        st.markdown(f"- {line}")

    g1, g2 = st.columns(2)
    rows = pd.DataFrame([state_to_row(s) for s in states])
    iss = issue_frame(states)
    with g1, st.container(border=True):
        st.markdown("**Feedback categories** · share of analysed reviews")
        st.altair_chart(CH.single_hue_bars(rows["feedback_type"], T.TEAL, "Reviews"), width="stretch")
    with g2, st.container(border=True):
        st.markdown("**Priority distribution** · P1 most urgent (Impact × Urgency matrix)")
        if iss.empty:
            st.caption("No operational issue extracted.")
        else:
            st.altair_chart(CH.priority_bars(iss["priority"]), width="stretch")
    g3, g4 = st.columns(2)
    with g3, st.container(border=True):
        st.markdown("**Most affected service-quality dimensions** · issues per dimension (provisional EGSQUAL)")
        if ins["dimensions"]:
            st.altair_chart(CH.single_hue_bars(pd.Series([d["name"].split(" (")[0] for d in ins["dimensions"] for _ in range(d["issues"])]),
                                               T.NAVY, "Issues"), width="stretch")
        else:
            st.caption("No operational issue extracted.")
    with g4, st.container(border=True):
        st.markdown("**Summary of actionable feedback** · Agent 3 decision per review")
        st.altair_chart(CH.single_hue_bars(rows["action"], C.AGENTS["Agent3"]["color"], "Reviews"), width="stretch")

    C.section("Most frequently reported issues", "grouped by topic and dimension · verbatim evidence")
    if ins["top_issues"]:
        _rtl_table(pd.DataFrame([{"Issue": t["issue"], "Dimension": t["dimension"], "Issues": t["count"], "P1/P2": t["urgent"],
                                  "Example evidence": t["examples"][0]["evidence"] if t["examples"] else ""}
                                 for t in ins["top_issues"][:6]]), ("Example evidence",))
    C.section("High-priority issues requiring attention", f"{len(ins['high_priority'])} issue(s) with priority P1 or P2")
    if ins["high_priority"]:
        _rtl_table(pd.DataFrame([{"Priority": r["priority"], "Application": r["app"], "Issue": r["description"], "Evidence": r["evidence"]}
                                 for r in ins["high_priority"][:8]]), ("Evidence",))
    else:
        st.caption("No P1/P2 issue in the analysed feedback.")
    C.section("Key improvement opportunities", "rule-based, each with the counts it rests on")
    for k, r in enumerate(ins["recommendations"], start=1):
        st.markdown(f"**{k}. {r['title']}**  \n{r['basis']}")
    _goto("insights", "Full Service Quality Insights and the executive PDF report")


# =========================================================================== #
# 2. Results & Evaluation
# =========================================================================== #
def page_results() -> None:
    from ui.eval_dashboard import page_eval_dashboard

    c = ctx()
    C.hero("Results & Evaluation", "Run the three agents on a sample, inspect every result and the agents' collaboration, "
           "and measure accuracy against human-verified labels.", [PROTOTYPE, "No fabricated scores"])
    t1, t2, t3 = st.tabs(["📋 Analysis results", "🔁 Agent collaboration trace", "🎯 Ground-truth evaluation"],
                         key="res_tabs", on_change="rerun")
    with t1:
        with st.container(border=True):
            a, b, d = st.columns([2, 1, 1])
            default_n = 200 if not c.llm.available else 10
            n = a.number_input("Analyse a random sample of reviews from the dataset", 1, 2000, default_n, step=10,
                               help="LLM mode makes about 3–6 API calls per review." if c.llm.available else "Offline mode: ~20 ms per review.")
            b.write("")
            b.write("")
            go = b.button("▶ Run sample analysis", type="primary", disabled=c.df is None, width="stretch")
            d.write("")
            d.write("")
            if d.button("Clear results", disabled=not store(), width="stretch"):
                st.session_state["store"] = {}
                st.session_state.pop("ins_pdf", None)
                st.rerun()
            if go:
                sample = c.df.sample(min(int(n), len(c.df)), random_state=None)
                remember(run_batch_with_progress([row_to_review(r) for _, r in sample.iterrows()], "Analysing"), "dashboard")
                st.rerun()
        states = [s for s in store().values() if s.agent3 is not None]
        if not states:
            C.empty_state("📋", "No results yet", "Run a sample above or analyse a review on the Agent 1 page.")
        else:
            agent_results(states, "res")
            table = pd.DataFrame([state_to_row(s) for s in states])
            C.section("All analysed reviews", f"{len(table)} row(s)")
            st.dataframe(table, hide_index=True, width="stretch", height=320)
            d1, d2 = st.columns(2)
            d1.download_button("⬇️ Results (CSV)", table.to_csv(index=False).encode("utf-8-sig"), "analysis_results.csv",
                               "text/csv", width="stretch")
            d2.download_button("⬇️ Full results with traces (JSON)",
                               json.dumps([state_to_dict(s) for s in states], ensure_ascii=False, indent=1).encode("utf-8"),
                               "analysis_results.json", "application/json", width="stretch")
    with t2:
        states = [s for s in store().values() if s.agent3 is not None]
        if not states:
            C.empty_state("🔁", "No trace yet", "Analyse a review first.")
        else:
            keys = {s.review.review_id: s for s in states}
            rid = st.selectbox("Review", list(keys), index=len(keys) - 1, key="trace_pick",
                               format_func=lambda k: f"{k} · {keys[k].review.app_name} · {keys[k].review.text[:70]}")
            s = keys[rid]
            C.workflow_diagram(s, llm=c.llm)
            req = sum(1 for m in s.messages if m.message_type == "REVIEW_REQUEST")
            st.caption(f"{len(s.messages)} message(s) · {req} review request(s) · path: {' → '.join(s.node_trace)}")
            C.message_timeline(s.messages)
            with st.expander(f"Decision log ({len(s.decision_log)}) and tool calls ({len(s.tool_calls)})"):
                log = pd.DataFrame([e.model_dump() for e in s.decision_log])
                st.dataframe(log.assign(agent=log["agent"].map(C.agent_name))[["step", "agent", "action", "detail"]],
                             hide_index=True, width="stretch")
                tl = pd.DataFrame([t.model_dump() for t in s.tool_calls])
                if not tl.empty:
                    st.dataframe(tl.assign(agent=tl["agent"].map(C.agent_name)), hide_index=True, width="stretch")
    with t3:
        page_eval_dashboard(show_hero=False)


# =========================================================================== #
# 3. Service Quality Insights (executive dashboard + PDF report)
# =========================================================================== #
def _rtl_table(df: pd.DataFrame, rtl_cols: tuple[str, ...]) -> None:
    """HTML table: Arabic evidence columns are set right-to-left; everything is escaped."""
    head = "".join(f"<th>{C.esc(str(c))}</th>" for c in df.columns)
    body = "".join("<tr>" + "".join(
        f'<td class="rtl">{C.esc(str(v))}</td>' if c in rtl_cols else f"<td>{C.esc(str(v))}</td>" for c, v in row.items()) + "</tr>"
        for _, row in df.iterrows())
    C.md(f'<div class="ins-table"><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>')


def page_insights() -> None:
    import insights as INS
    import report_pdf

    C.hero("Service Quality Insights", "Executive view of the analysed beneficiary feedback: affected service-quality dimensions, "
           "frequent issues, urgent feedback and Agent 3's recommended improvements.", [INS.PROTOTYPE_LABEL])
    states = [s for s in store().values() if s.agent3 is not None]
    ins = INS.compute_insights(states)
    cov = ins["coverage"]

    with st.container(border=True):
        a, b = st.columns([3, 2])
        a.markdown("**Executive report (PDF)** · 10 sections, dark-green identity, Arabic evidence right-to-left, "
                   "built only from the results below.")
        if not states:
            a.caption("Disabled: no reviews have been analysed in this session yet. Run a sample on the Dashboard "
                      "or analyse a review first; the report never uses invented figures.")
        if b.button("📄 Generate Executive Report", type="primary", disabled=not states, width="stretch", key="ins_gen"):
            with st.spinner("Building the PDF…"):
                st.session_state["ins_pdf"] = (report_pdf.build_executive_report(ins), cov["reviews"], cov["analysis_date"])
        pdf = st.session_state.get("ins_pdf")
        if pdf and states:
            data, n, day = pdf
            if n != cov["reviews"]:
                b.caption(f"⚠️ This PDF covers {n} reviews; {cov['reviews']} are analysed now. Generate it again to include them.")
            b.download_button("⬇️ Download the PDF", data, f"executive_report_{day}.pdf", "application/pdf",
                              width="stretch", key="ins_dl")

    if not states:
        C.empty_state("📊", "No analysed feedback yet", "Every number on this page is computed from reviews actually analysed "
                      "in this session. Run a sample on the Dashboard or analyse a review first.")
        return

    urgent = sum(n for p, n, _ in ins["priority"] if p in INS.URGENT)
    hc = sum(n for s_, n, _ in ins["severity"] if s_ in ("High", "Critical"))
    C.kpi_row([("Reviews analysed", cov["reviews"], f"{cov['apps']} application(s)", "navy"),
               ("Service issues", cov["issues"], "evidence-backed"),
               ("High / Critical", hc, "service blocked or disrupted", "crit" if hc else ""),
               ("P1 / P2", urgent, "require attention first", "warn" if urgent else "")])
    C.section("Executive summary", "sentences built only from the counts below")
    for line in ins["summary"]:
        st.markdown(f"- {line}")

    C.section("Top affected service-quality dimensions", "with the provisional EGSQUAL dimension (Aljukhadar et al., 2022)")
    if ins["dimensions"]:
        g1, g2 = st.columns([2, 3])
        with g1, st.container(border=True):
            dims = pd.Series([d["name"].split(" (")[0] for d in ins["dimensions"] for _ in range(d["issues"])])
            st.altair_chart(CH.single_hue_bars(dims, T.TEAL, "Issues"), width="stretch")
        with g2:
            st.dataframe(pd.DataFrame([{"dimension": d["name"], "EGSQUAL (provisional)": d["egsqual"] or "—", "issues": d["issues"],
                                        "share %": d["share"], "High/Critical": d["high_critical"], "P1/P2": d["urgent"],
                                        "most affected app": f"{d['top_apps'][0][0]} ({d['top_apps'][0][1]})" if d["top_apps"] else "—"}
                                       for d in ins["dimensions"]]), hide_index=True, width="stretch")
    else:
        st.caption("No operational issue was extracted.")

    C.section("Most frequently reported issues", "grouped by topic and dimension · verbatim evidence")
    if ins["top_issues"]:
        _rtl_table(pd.DataFrame([{"Issue": t["issue"], "Dimension": t["dimension"], "Issues": t["count"], "P1/P2": t["urgent"],
                                  "Example evidence": t["examples"][0]["evidence"] if t["examples"] else "",
                                  "Review": t["examples"][0]["review_id"] if t["examples"] else ""} for t in ins["top_issues"]]),
                   ("Example evidence",))
    else:
        st.caption("No operational issue was extracted.")

    C.section("Feedback across applications", "per application, from analysed reviews only")
    st.dataframe(pd.DataFrame([{"application": x["app"], "reviews": x["reviews"], "issues": x["issues"],
                                "High/Critical": x["high_critical"], "P1/P2": x["urgent"], "most affected dimension": x["top_dimension"],
                                "most frequent issue": x["top_issue"]} for x in ins["by_app"]]), hide_index=True, width="stretch")
    sev_c = pd.Series([s_ for s_, n, _ in ins["severity"] for _ in range(n)])
    a_, b_ = st.columns(2)
    with a_, st.container(border=True):
        st.markdown("**Severity distribution** · issues (rules R1–R4)")
        if len(sev_c):
            st.altair_chart(CH.severity_bars(sev_c), width="stretch")
    with b_, st.container(border=True):
        st.markdown("**Priority distribution** · issues (Impact × Urgency matrix)")
        pr_c = pd.Series([p for p, n, _ in ins["priority"] for _ in range(n)])
        if len(pr_c):
            st.altair_chart(CH.priority_bars(pr_c), width="stretch")
    recurring = [t for t in ins["top_issues"] if t["reviews"] >= 2]
    if recurring:
        st.markdown("**Recurring issues** (reported in two or more reviews): "
                    + "; ".join(f"{t['issue']} — {t['reviews']} reviews in {', '.join(t['apps'])}" for t in recurring[:5]))

    C.section("High-priority feedback requiring attention", f"{len(ins['high_priority'])} issue(s) with priority P1 or P2")
    if ins["high_priority"]:
        _rtl_table(pd.DataFrame([{"Priority": r["priority"], "Severity": r["severity"], "Application": r["app"],
                                  "Issue": r["description"], "Evidence": r["evidence"], "Review": r["review_id"]}
                                 for r in ins["high_priority"][:25]]), ("Evidence",))
        if len(ins["high_priority"]) > 25:
            st.caption(f"Showing 25 of {len(ins['high_priority'])}; the full list is in the Dashboard CSV export.")
    else:
        st.caption("No P1/P2 issue in the analysed feedback.")

    C.section("Recommended service improvements", "from Agent 3's decisions; recommendations state their basis")
    a, b = st.columns([2, 3])
    with a, st.container(border=True):
        st.markdown("**Agent 3 actions** (reviews)")
        st.dataframe(pd.DataFrame(ins["actions"], columns=["action", "reviews", "share %"]), hide_index=True, width="stretch")
    with b, st.container(border=True):
        st.markdown("**Management recommendations**")
        for k, r in enumerate(ins["recommendations"], start=1):
            st.markdown(f"**{k}. {r['title']}**  \n{r['basis']}")
    if ins["improvements"]:
        st.dataframe(pd.DataFrame([{"action": g["action"], "application": g["app"], "dimension": g["dimension"],
                                    "handling team": g["team"], "tickets": g["tickets"], "P1/P2": g["urgent"],
                                    "example ticket": g["example_title"]}
                                   for g in ins["improvements"]]), hide_index=True, width="stretch")
    st.caption("🛡️ Tickets and escalations are simulated; escalation rules are research-defined. Nothing is sent to any government system.")


# =========================================================================== #
# 4. Methodology & rules
# =========================================================================== #
def _egsqual_section() -> None:
    """EGSQUAL subsection: reference, verification status, crosswalk, Arabic examples, mapping rules, limitations."""
    kb, ref = tools.egsqual_kb(), tools.egsqual_reference()
    cit, ver = ref["citation"], ref["verification"]
    C.section("EGSQUAL — service-quality dimensions (Agent 1)", "primary reference and how it is used")
    C.md(f'<div class="card"><h4>Primary reference</h4><div>{C.esc(cit["authors"])} ({cit["year"]}). <i>{C.esc(cit["title"])}</i>. '
         f'{C.esc(cit["journal"])}, {C.esc(cit["volume"])}, {C.esc(cit["article"])}. '
         f'DOI: <a href="{C.esc(cit["doi_url"])}" target="_blank">{C.esc(cit["doi"])}</a></div>'
         f'<div class="small" style="margin-top:6px">{C.esc(cit["authors_note"])}</div></div>')
    if not ver["full_text_accessed"]:
        st.warning("**Verification status: full text NOT accessed.** " + ver["method"] + "  \n**Action needed:** " + ver["action_needed"])
    a, b = st.columns(2)
    with a:
        st.markdown("**Verified (from the authors' abstract, via search summary)**")
        for x in ver["verified_from_search_summary"]:
            st.markdown(f"- {x}")
    with b:
        st.markdown("**Not verified — not used as fact**")
        for x in ver["not_verified"]:
            st.markdown(f"- {x}")
    st.markdown("**The seven EGSQUAL dimensions and the prototype's categories (provisional crosswalk)**")
    st.dataframe(pd.DataFrame([{"EGSQUAL dimension": d["name"], "name status": d["name_status"], "definition": d["definition_status"],
                                "prototype categories": ", ".join(kb[k]["name_en"] for k in d["prototype_categories"]) or "— none yet"}
                               for d in ref["egsqual_dimensions"]]), hide_index=True, width="stretch")
    st.caption(ref["crosswalk_status"])

    st.markdown("**Operational categories used by Agent 1**")
    st.dataframe(pd.DataFrame([{"key": k, "category": v["name_en"], "arabic": v["name_ar"], "definition": v["definition"],
                                "EGSQUAL (provisional)": v.get("egsqual_candidate") or "—", "source": v["source"]}
                               for k, v in kb.items()]), hide_index=True, width="stretch")
    with st.expander("Arabic feedback examples for each category (real reviews from the research dataset)"):
        for k, v in kb.items():
            for e in v.get("examples_ar", []):
                C.md(f'<div class="small" style="margin-top:8px"><b>{C.esc(v["name_en"])}</b> · {C.esc(e["review_id"])} · {C.esc(e["app_id"])}</div>'
                     f'<div class="review-box"><div class="rtl">{C.esc(e["text"])}</div></div>')

    st.markdown("**Mapping rules used by Agent 1**")
    for x in [
        "Each review is split into independent issues; every issue gets exactly one category, so a review with several issues "
        "can carry several dimensions.",
        "In LLM mode, the model receives the category definitions from the knowledge base and must return a category key, a "
        "one-sentence reason and a verbatim Arabic evidence quote. In offline mode, the topic lexicon assigns the category.",
        "The evidence verifier checks that the quote occurs in the review; an issue without supporting evidence is rejected.",
        "A category key that does not exist is corrected by the knowledge-base keyword search; if no keyword matches, the issue "
        "is set to Unclear / Unmapped instead of being guessed.",
        "When the review is ambiguous and the issue is a generic malfunction with no category keyword in its evidence, the "
        "issue is set to Unclear / Unmapped.",
        "Each issue shows its category, the provisional EGSQUAL dimension, the Arabic evidence and the reason for the choice.",
    ]:
        st.markdown(f"- {x}")
    st.markdown("**How the dimensions are used by Agents 2 and 3**")
    for x in [
        "Agent 2 receives each issue's category as context, but severity and priority are set only by rules R1–R4 and the "
        "P-rules from impact facts; a category never raises or lowers severity.",
        "Agent 1's duplicate check treats two issues as the same problem only when they share a category.",
        "Agent 3 writes the categories into the simulated ticket so it can be routed to the owner of that service aspect; the "
        "action itself depends on the issue type and whether the service could be completed.",
        "The Service Quality Insights page and the executive report aggregate issues, severity and priority by category.",
    ]:
        st.markdown(f"- {x}")
    st.markdown("**Adaptation of EGSQUAL**")
    for x in ref["adaptation"]:
        st.markdown(f"- {x}")
    st.markdown("**Limitations**")
    for x in ref["limitations"]:
        st.markdown(f"- {x}")


def page_methodology() -> None:
    import json as _json

    C.hero("Methodology & Rules", "How GovFeedback AI decides. Published frameworks inform the design; every scoring rule, "
           "threshold and SLA value below is research-defined and editable in knowledge_base/.",
           [PROTOTYPE, "Not official DGA rules", "No compliance claimed"])
    fw = _json.loads((config.KB_DIR / "frameworks.json").read_text(encoding="utf-8"))
    C.md('<div class="card"><h4>Architecture and collaboration</h4><div class="small">'
         '<b>Agent 1</b> extracts independent issues with verbatim evidence and maps each to a service-quality category. '
         '<b>Agent 2</b> assesses impact and urgency, sets severity by rules R1–R4 and priority by the Impact × Urgency matrix, and '
         '<b>critiques</b> Agent 1: weak evidence, a type conflict, unclear ambiguity or duplicates trigger a review request; Agent 1 '
         'confirms or revises, at most two rounds. <b>Agent 3</b> decides the action, routes the ticket, applies simulated SLA '
         'targets, checks complaint history and escalation rules E1–E5, and drafts an Arabic reply. A shared state records every '
         'message, tool call and decision.</div></div>')

    C.section("Published frameworks vs research-defined rules", "what is borrowed, and what is this project's own design")
    st.warning(fw["_about"])
    _rtl_table(pd.DataFrame([{"Framework": f["name"], "Published concept (summary)": f["published_concept"],
                              "How it informs the prototype": f["used_for"], "Research-defined (custom)": f["custom"]}
                             for f in fw["frameworks"]]), ())

    _egsqual_section()

    sr = tools.severity_rules_kb()
    mk = tools.priority_matrix_kb()
    C.section("Agent 2 — severity rules and the Impact × Urgency matrix", "ITIL 4-informed; matrix cells are research-defined")
    st.info(sr["principle"])
    st.dataframe(pd.DataFrame(sr["severity_rules"]), hide_index=True, width="stretch")
    a, b = st.columns([3, 2])
    with a:
        st.dataframe(pd.DataFrame(mk["impact_rules"] + mk["urgency_rules"] + [{"id": mk["verification_cap"]["id"], "level": "cap",
                                                                              "rule": mk["verification_cap"]["rule"]}]),
                     hide_index=True, width="stretch")
    with b:
        C.matrix_grid(ctx().matrix, caption="Current matrix (default from knowledge_base/impact_urgency_matrix.json; "
                      "can be changed in Settings).")

    C.section("Agent 3 — simulated SLA targets and handling teams", "research-defined simulation, not official service levels")
    st.warning(tools.sla_targets_kb()["_about"])
    a, b = st.columns(2)
    a.dataframe(pd.DataFrame([{"priority": p, "response (h)": v["response_hours"], "resolution (h)": v["resolution_hours"]}
                              for p, v in ctx().sla_targets.items()]), hide_index=True, width="stretch")
    ht = tools.handling_teams_kb()
    b.dataframe(pd.DataFrame([{"rule": f"{kind}: {k}", "team": v} for kind in ("by_action", "by_topic", "by_dimension")
                              for k, v in ht[kind].items()]), hide_index=True, width="stretch", height=300)

    er = tools.escalation_rules_kb()
    C.section("Agent 3 — escalation eligibility rules", "simulated; human approval always required")
    st.warning("Proposed, configurable research rules — not official rules of the DGA or the AMER / Balaagh Digital service. "
               "Escalation is simulated only.")
    st.dataframe(pd.DataFrame(er["rules"]), hide_index=True, width="stretch")
    st.caption(f"Minimum severity {er['min_severity_for_escalation']} · default complaint SLA {er['default_response_sla_days']} days "
               f"({er['default_sla_note']}) · human approval required: {er['require_human_approval']}")

    C.section("Human-in-the-loop requirements")
    for x in fw["human_in_the_loop"]:
        st.markdown(f"- {x}")
