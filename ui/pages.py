"""Dashboard pages. Shared context (LLM client, workflow, dataset) is put in
st.session_state["ctx"] by app.py. Results of every run are kept in a session
store so the overview dashboard reflects only analyses that really ran."""
from __future__ import annotations

import io
import json
from dataclasses import dataclass
from datetime import date
from typing import Optional

import pandas as pd
import streamlit as st

import evaluation
import tools
from data_loader import row_to_review
from scenarios import get_scenarios, verify_against_dataset
from schemas import ReviewInput, WorkflowState
from ui import charts as CH
from ui import components as C
from ui import theme as T
from ui.result_view import render_result, run_live
from workflow import state_to_dict, state_to_row


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
        C.empty_state("📊", "No analysis yet", "Run a sample above, analyse a single review, or run a batch — the KPIs and charts "
                      "are computed from those real results only.")
        return
    agent_results(states, "dash")


# =========================================================================== #
# 2. Analyse feedback
# =========================================================================== #
EXAMPLES = {
    "Absher — renewal error, fees deducted": ("Absher", 2, "ابي اجدد الاقامة للعامل والتطبيق يقول حدث خطأ غير متوقع، حاولت اكثر من مره ونفس المشكله، والرسوم انخصمت من الحساب"),
    "Nafath — widespread login failure": ("Nafath", 1, "الكل عندهم نفس المشكله نفاذ ما يرسل الاشعار للجوال وما نقدر ندخل اي منصة حكومية"),
    "Sehhaty — booking not fulfilled + freezes": ("Sehhaty", 1, "حجزت موعد في المركز الصحي من التطبيق وطلع لي تم الحجز بس لما رحت قالوا ما فيه موعد باسمك، وفوق كذا التطبيق يعلق كل ما افتح صفحة المواعيد"),
    "Tawakkalna — praise only": ("Tawakkalna", 5, "التطبيق ممتاز وسهل الاستخدام الله يعطيكم العافية"),
}


def page_analyze() -> None:
    c = ctx()
    C.hero("Analyse Beneficiary Feedback", "Type an Arabic review, pick one from the dataset or upload a CSV — then watch the three agents work.",
           ["RTL Arabic", "Saudi dialect", "Live agent trace"])
    C.mode_banner(c.llm)
    C.agents_overview(compact=True)
    mode = st.segmented_control("Input", ["✍️ Type a review", "📚 Pick from dataset", "📤 Upload CSV"],
                                default="✍️ Type a review", key="an_mode")
    review: Optional[ReviewInput] = None
    if mode == "✍️ Type a review":
        ex = st.selectbox("Load an example (written by the researcher, not from the dataset)", ["—"] + list(EXAMPLES), key="an_ex")
        if ex != "—" and st.session_state.get("an_ex_loaded") != ex:
            app, rating, text = EXAMPLES[ex]
            st.session_state.update({"an_text": text, "an_app": app, "an_rating": rating, "an_ex_loaded": ex})
        txt = st.text_area("Arabic review", key="an_text", height=110, placeholder="اكتب مراجعة المستفيد هنا…")
        a, b = st.columns(2)
        st.session_state.setdefault("an_app", "CUSTOM")
        app = a.text_input("Application", key="an_app")
        rating = b.selectbox("Star rating", [None, 1, 2, 3, 4, 5], key="an_rating")
        if txt and txt.strip():
            review = ReviewInput(review_id="TYPED-1", text=txt.strip(), app_id=app or "CUSTOM", app_name=app or "CUSTOM", rating=rating)
    elif mode == "📚 Pick from dataset":
        sub = review_picker("an")
        if sub is not None and not sub.empty:
            opts = sub.head(500)
            idx = st.selectbox("Review", opts.index, key="an_pick",
                               format_func=lambda i: f"{opts.at[i, 'review_id']} · {opts.at[i, 'app_id']} · ★{opts.at[i, 'rating']} · {opts.at[i, 'text'][:80]}")
            review = row_to_review(opts.loc[idx])
    else:
        up = st.file_uploader("Upload a reviews CSV (UTF-8; needs a review-text column such as 'Content')", type=["csv"], key="an_upload")
        if up is not None:
            st.session_state["dataset_bytes"] = up.getvalue()
            st.session_state["dataset_name"] = up.name
            st.success(f"Loaded {up.name}. It is now the active dataset — use “Pick from dataset”, Batch or the Dashboard.")
            st.rerun()
        if c.info:
            st.caption(f"Active dataset: {st.session_state.get('dataset_name', 'data/reviews.csv')} · {c.info['rows_available']:,} unique reviews")

    if review is not None:
        C.section("Original review", "right-to-left")
        C.md(f'<div class="review-box"><div class="rtl">{C.esc(review.text)}</div></div>')
    run = st.button("▶ Run Multi-Agent Analysis", type="primary", disabled=review is None, key="an_run")
    if run and review is not None:
        state = run_live(c.workflow, review, c.ref_date, c.llm)
        st.session_state["an_state"] = state
        remember([state], "single")
    if "an_state" in st.session_state:
        render_result(st.session_state["an_state"], c.llm, "an")


# =========================================================================== #
# 3. Scenarios
# =========================================================================== #
def page_scenarios() -> None:
    c = ctx()
    C.hero("Test Scenarios", "Eight real reviews from the dataset and one clearly labelled synthetic case for the escalation logic.",
           ["Clear technical issue", "Multiple issues", "Ambiguous", "Praise + suggestion", "High impact", "Escalation"])
    C.mode_banner(c.llm)
    scen = get_scenarios()
    ver = verify_against_dataset()
    keys = [s.key for s in scen]
    cols = st.columns(3)
    for i, s in enumerate(scen):
        with cols[i % 3]:
            src = "🧪 SYNTHETIC" if s.synthetic else (f"✅ dataset row {s.original_row}" if ver["checked"].get(s.key) else f"dataset row {s.original_row}")
            C.md(f'<div class="card" style="min-height:150px"><div style="display:flex;justify-content:space-between">'
                 f'{C.pill(s.key, "navy")}<span class="small">{C.esc(src)}</span></div><h4 style="margin-top:6px">{C.esc(s.title)}</h4>'
                 f'<div class="small">{C.esc(s.purpose)}</div></div>')
    pick = st.segmented_control("Choose a scenario", keys, default=keys[0], key="sc_pick")
    sc = next(s for s in scen if s.key == pick) if pick else scen[0]
    if sc.synthetic:
        st.warning(f"🧪 Synthetic scenario: the review and complaint record SYN-2026-0417 are artificial. SLA reference date fixed at {sc.reference_date}.")
    C.md(f'<div class="review-box"><div class="rtl">{C.esc(sc.review.text)}</div></div>')
    if sc.expect:
        st.caption("Design expectations (what this scenario should demonstrate — not ground truth): "
                   + ", ".join(f"{k}={v}" for k, v in sc.expect.items()))
    if st.button(f"▶ Run scenario {sc.key}", type="primary", key="sc_run"):
        state = run_live(c.workflow, sc.review, sc.reference_date or c.ref_date, c.llm)
        st.session_state["sc_state"] = (sc.key, state)
        remember([state], "scenario")
        checks = evaluation.behavioural_checks(state, sc.expect)
        st.session_state["sc_checks"] = (sc.key, checks)
    if st.session_state.get("sc_state", (None,))[0] == sc.key:
        checks = st.session_state.get("sc_checks", (None, []))[1]
        if checks:
            ok = sum(1 for x in checks if x.passed)
            app_ = sum(1 for x in checks if x.passed is not None)
            (st.success if ok == app_ else st.error)(f"Behavioural checks for {sc.key}: {ok}/{app_} passed")
        render_result(st.session_state["sc_state"][1], c.llm, f"sc_{sc.key}")


# =========================================================================== #
# 4. Batch
# =========================================================================== #
def page_batch() -> None:
    c = ctx()
    C.hero("Batch Analysis", "Run the agents on many reviews, compare outcomes and drill into any single result.", ["Filters", "Charts", "CSV / JSON export"])
    C.mode_banner(c.llm)
    sub = review_picker("bt")
    if sub is None:
        return
    a, b = st.columns([1, 3])
    how = a.radio("Selection", ["First N", "Random N", "Pick reviews"], key="bt_how")
    if how == "Pick reviews":
        opts = sub.head(500)
        picked = b.multiselect("Reviews", opts.index, key="bt_picked",
                               format_func=lambda i: f"{opts.at[i, 'review_id']} · {opts.at[i, 'text'][:70]}")
        batch = opts.loc[picked]
    else:
        n = b.number_input("N", 1, 2000, 20, key="bt_n")
        batch = sub.head(int(n)) if how == "First N" else sub.sample(min(int(n), len(sub)), random_state=42)
    st.caption(f"{len(batch)} review(s) selected" + (" · LLM mode: about 2–4 API calls per review" if c.llm.available else ""))
    if st.button("▶ Run batch analysis", type="primary", disabled=batch.empty, key="bt_run"):
        states = run_batch_with_progress([row_to_review(r) for _, r in batch.iterrows()], "Batch")
        st.session_state["batch_states"] = states
        remember(states, "batch")
    states = st.session_state.get("batch_states")
    if not states:
        return
    rows = agent_results(states, "batch", show_rating_chart=False)
    C.section("All results")
    st.dataframe(rows[["review_id", "app_id", "rating", "review_text", "feedback_type", "n_issues", "overall_severity",
                       "overall_priority", "revision_rounds", "action", "escalation_status"]], hide_index=True, width="stretch", height=320)
    d1, d2 = st.columns(2)
    d1.download_button("⬇️ Batch results (CSV)", rows.to_csv(index=False).encode("utf-8-sig"), "batch_results.csv", "text/csv", width="stretch")
    d2.download_button("⬇️ Batch results (JSON)", json.dumps([state_to_dict(s) for s in states], ensure_ascii=False, indent=1),
                       "batch_results.json", "application/json", width="stretch")
    C.section("Inspect one result")
    pick = st.selectbox("Result", range(len(states)), key="bt_inspect",
                        format_func=lambda i: f"{states[i].review.review_id} · {states[i].review.text[:80]}")
    render_result(states[pick], c.llm, f"bt_{pick}")


# =========================================================================== #
# 5. Evaluation
# =========================================================================== #
LABEL_COLUMNS = ["feedback_type", "severity", "priority", "action", "escalation_status", "dimensions"]


def page_evaluation() -> None:
    c = ctx()
    C.hero("Evaluation", "Rule-compliance checks on the test scenarios, and Precision / Recall / F1 only against verified reference labels.",
           ["No fabricated scores", "Scenario-based", "Optional labelled evaluation"])
    C.mode_banner(c.llm)

    C.section("1 · Behavioural checks on the scenarios", "rule compliance & evidence support — not accuracy")
    if st.button("▶ Run all scenarios and evaluate", type="primary", key="ev_run"):
        res = []
        bar = st.progress(0.0)
        scen = get_scenarios()
        for k, s in enumerate(scen):
            st_ = c.workflow.run(s.review, reference_date=s.reference_date or c.ref_date)
            res.append((s.key, st_, s.expect))
            bar.progress((k + 1) / len(scen), text=f"{s.key}")
        bar.empty()
        remember([r[1] for r in res], "scenario")
        st.session_state["ev_checks"] = evaluation.checks_table(res)
    chk = st.session_state.get("ev_checks")
    if chk is None:
        C.empty_state("🧪", "Not run yet", "Click the button to run the scenarios and their checks.")
    else:
        s = evaluation.summary(chk)
        C.kpi_row([("Checks passed", f"{s['checks_passed']} / {s['applicable']}", "applicable checks", ""),
                   ("Checks failed", s["checks_failed"], "", "crit" if s["checks_failed"] else ""),
                   ("Not applicable", s["not_applicable"], "e.g. no issues to verify", "navy"),
                   ("Scenarios", chk["scenario"].nunique(), "8 real + 1 synthetic", "navy")])
        a, b = st.columns([2, 3])
        with a:
            st.altair_chart(CH.checks_by_scenario(chk), width="stretch")
        with b:
            only = st.segmented_control("Show", ["All", "FAIL", "PASS", "N/A"], default="All", key="ev_filter")
            view = chk if (not only or only == "All") else chk[chk["result"] == only]
            st.dataframe(view, hide_index=True, width="stretch", height=360)

    C.section("2 · Precision / Recall / F1", "computed only from verified reference labels you upload")
    st.markdown("Upload a CSV with a `review_id` column (e.g. `R00705`) and any of these label columns: "
                + ", ".join(f"`{x}`" for x in LABEL_COLUMNS) + ". `dimensions` is semicolon-separated. Labels must be "
                "**verified by a human annotator**; the dataset itself contains none.")
    tmpl = pd.DataFrame({"review_id": ["R00705", "R02464"], **{col: ["", ""] for col in LABEL_COLUMNS}})
    st.download_button("⬇️ Label template (CSV)", tmpl.to_csv(index=False).encode("utf-8-sig"), "reference_labels_template.csv", "text/csv")
    up = st.file_uploader("Reference labels CSV", type=["csv"], key="ev_labels")
    if up is None:
        C.empty_state("🔒", "No reference labels → no performance metrics", "Precision, Recall and F1 are never estimated or "
                      "simulated. Upload verified labels to compute them.")
        return
    try:
        ref = pd.read_csv(io.BytesIO(up.getvalue()), encoding="utf-8-sig", dtype=str)
    except Exception as e:  # noqa: BLE001
        st.error(f"Could not read the CSV: {e}")
        return
    if "review_id" not in ref.columns:
        st.error("The file needs a `review_id` column.")
        return
    analysed = {s.review.review_id: s for s in store().values()}
    missing = [rid for rid in ref["review_id"].astype(str) if rid not in analysed]
    if missing:
        can = c.df is not None and c.df["review_id"].isin(missing).any()
        absent = [rid for rid in missing if c.df is None or rid not in set(c.df["review_id"])]
        st.info(f"{len(missing)} labelled review(s) have not been analysed in this session yet."
                + (f" {len(absent)} of them are not in the loaded dataset (unknown id, or removed as a duplicate of an "
                   f"earlier review from the same app): {', '.join(absent[:10])}." if absent else ""))
        if can and st.button(f"▶ Run the agents on the {int(c.df['review_id'].isin(missing).sum())} labelled review(s)", type="primary"):
            todo = c.df[c.df["review_id"].isin(missing)]
            remember(run_batch_with_progress([row_to_review(r) for _, r in todo.iterrows()], "Labelled reviews"), "labelled")
            st.rerun()
    states = list({s.review.review_id: s for s in store().values()}.values())
    try:
        res = evaluation.prf_from_reference(ref, states)
    except ValueError as e:
        st.error(str(e))
        return
    if not res:
        C.empty_state("⚠️", "Nothing to score", "No overlap between the labelled review_ids / label columns and the analysed results.")
        return
    for field, table in res.items():
        with st.container(border=True):
            st.markdown(f"**{field}** · {int(table['support'].iloc[-1])} labelled item(s)")
            body = table[~table["label"].isin(["MACRO AVG", "ACCURACY", "MICRO AVG"])]
            macro_f1 = body["f1"].mean() if not body.empty else 0.0
            kp = [("Macro F1", f"{macro_f1:.3f}", "mean over classes"),
                  ("Macro precision", f"{body['precision'].mean():.3f}", ""),
                  ("Macro recall", f"{body['recall'].mean():.3f}", "")]
            if (table["label"] == "ACCURACY").any():
                kp.append(("Accuracy", f"{float(table.loc[table['label'] == 'ACCURACY', 'f1'].iloc[0]):.3f}", "exact match", "navy"))
            if (table["label"] == "MICRO AVG").any():
                kp.append(("Micro F1", f"{float(table.loc[table['label'] == 'MICRO AVG', 'f1'].iloc[0]):.3f}", "multi-label", "navy"))
            C.kpi_row(kp)
            st.altair_chart(CH.prf_bars(table), width="stretch")
            st.dataframe(table, hide_index=True, width="stretch")


# =========================================================================== #
# 6. Methodology & rules
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
