"""Dashboard pages. Shared context (LLM client, workflow, dataset) is put in
st.session_state["ctx"] by app.py. Results of every run are kept in a session
store so the overview dashboard reflects only analyses that really ran."""
from __future__ import annotations

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
    if ctx().llm.fatal_error:
        st.session_state["llm_fatal"] = ctx().llm.fatal_error
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


PAGES: dict = {}  # name -> st.Page, filled by app.py so pages can link to each other
PROTOTYPE = config.PROTOTYPE_LABEL


def presentation() -> bool:
    """Management Presentation Mode: hides developer diagnostics, keeps limitations and simulation labels."""
    return bool(st.session_state.get("cfg_presentation", False))


def _goto(page: str, label: str) -> None:
    if page in PAGES:
        st.page_link(PAGES[page], label=label, icon=":material/arrow_forward:")


def analysed() -> list[WorkflowState]:
    return [s for s in store().values() if s.agent3 is not None]


def _rtl_table(df: pd.DataFrame, rtl_cols: tuple[str, ...]) -> None:
    """HTML table: Arabic evidence columns are set right-to-left; everything is escaped."""
    head = "".join(f"<th>{C.esc(str(c))}</th>" for c in df.columns)
    body = "".join("<tr>" + "".join(
        f'<td class="rtl">{C.esc(str(v))}</td>' if c in rtl_cols else f"<td>{C.esc(str(v))}</td>" for c, v in row.items()) + "</tr>"
        for _, row in df.iterrows())
    C.md(f'<div class="ins-table"><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>')


# =========================================================================== #
# 1. Data & Run Analysis (starting page)
# =========================================================================== #
EXAMPLES = {
    "Absher — renewal error, fees deducted": ("Absher", 2, "ابي اجدد الاقامة للعامل والتطبيق يقول حدث خطأ غير متوقع، حاولت اكثر من مره ونفس المشكله، والرسوم انخصمت من الحساب"),
    "Nafath — widespread login failure": ("Nafath", 1, "الكل عندهم نفس المشكله نفاذ ما يرسل الاشعار للجوال وما نقدر ندخل اي منصة حكومية"),
    "Sehhaty — booking not fulfilled + freezes": ("Sehhaty", 1, "حجزت موعد في المركز الصحي من التطبيق وطلع لي تم الحجز بس لما رحت قالوا ما فيه موعد باسمك، وفوق كذا التطبيق يعلق كل ما افتح صفحة المواعيد"),
    "Tawakkalna — praise only": ("Tawakkalna", 5, "التطبيق ممتاز وسهل الاستخدام الله يعطيكم العافية"),
}


def _status_counts(states: list[WorkflowState]) -> dict:
    from workflow import execution_status
    out = {"LLM": 0, "Rule-based fallback": 0, "Rule-based (offline)": 0, "Failed": 0}
    for s in states:
        out[execution_status(s)] += 1
    return out


def _single_review_input(c) -> None:
    from scenarios import get_scenarios

    scen = {f"Test scenario {s.key} — {s.title}" + (" (synthetic)" if s.synthetic else ""): s for s in get_scenarios()}
    ex = st.selectbox("Load an example or a test scenario (S6: review loop · S8: simulated escalation)",
                      ["—"] + list(EXAMPLES) + list(scen), key="an_ex")
    if ex != "—" and st.session_state.get("an_ex_loaded") != ex:
        if ex in scen:
            r = scen[ex].review
            st.session_state.update({"an_text": r.text, "an_app": r.app_id, "an_rating": r.rating})
        else:
            app, rating, text = EXAMPLES[ex]
            st.session_state.update({"an_text": text, "an_app": app, "an_rating": rating})
        st.session_state["an_ex_loaded"] = ex
    txt = st.text_area("Arabic review", key="an_text", height=90, placeholder="اكتب مراجعة المستفيد هنا…")
    a, b = st.columns(2)
    st.session_state.setdefault("an_app", "CUSTOM")
    app = a.text_input("Application", key="an_app")
    rating = b.selectbox("Star rating", [None, 1, 2, 3, 4, 5], key="an_rating")
    review, ref_date = None, c.ref_date
    sc = scen.get(ex)
    if sc is not None and txt.strip() == sc.review.text.strip():  # unchanged scenario: keep its id, flags and SLA date
        review, ref_date = sc.review, sc.reference_date or c.ref_date
        if sc.synthetic:
            st.caption(f"🧪 Synthetic scenario: the review and its complaint record are artificial; SLA date fixed at {sc.reference_date}.")
    elif txt and txt.strip():
        n = st.session_state.get("typed_n", 0) + 1
        review = ReviewInput(review_id=f"TYPED-{n}", text=txt.strip(), app_id=app or "CUSTOM", app_name=app or "CUSTOM", rating=rating)
    if st.button("▶ Run Multi-Agent Analysis on this review", disabled=review is None, key="an_run"):
        from ui.result_view import run_live
        if review.review_id.startswith("TYPED-"):
            st.session_state["typed_n"] = st.session_state.get("typed_n", 0) + 1
        state = run_live(c.workflow, review, ref_date, c.llm)
        remember([state], "single")
        st.session_state["current_key"] = f"single:{state.review.review_id}"
        if c.llm.fatal_error:
            st.session_state["llm_fatal"] = c.llm.fatal_error
        st.session_state["last_run"] = _status_counts([state])


def page_data_run() -> None:
    c = ctx()
    C.hero(config.APP_TITLE_EN, "Upload or select the beneficiary reviews, then run the three cooperating agents.",
           [PROTOTYPE], title_ar=config.APP_TITLE_AR)

    C.section("1 · Dataset", "the only place where data is loaded")
    a, b = st.columns([1, 2])
    src = a.radio("Source", ["Existing dataset", "Upload a CSV file"], key="data_src", label_visibility="collapsed")
    if src == "Upload a CSV file":
        up = b.file_uploader("Reviews CSV (UTF-8; needs a review-text column such as 'Content')", type=["csv"], key="data_upload")
        if up is not None and st.session_state.get("dataset_name") != up.name:
            st.session_state["dataset_bytes"] = up.getvalue()
            st.session_state["dataset_name"] = up.name
            st.rerun()
    elif st.session_state.get("dataset_bytes"):
        if b.button("Use the existing dataset (data/reviews.csv) again"):
            st.session_state.pop("dataset_bytes", None)
            st.session_state.pop("dataset_name", None)
            st.rerun()
    if c.info:
        b.caption(f"Loaded: **{st.session_state.get('dataset_name', 'data/reviews.csv')}**")
        C.kpi_row([("Total reviews", f"{c.info['rows_in_file']:,}", "rows in the file", "navy"),
                   ("Unique reviews", f"{c.info['rows_available']:,}", f"{c.info['duplicates_dropped']} duplicates removed", "navy"),
                   ("Applications", c.info["apps"], "government apps", "navy"),
                   ("Average rating", f"{c.df['rating'].mean():.2f} ★", "full dataset", "navy")])
    else:
        C.empty_state("📁", "No dataset loaded", "Upload a CSV, or place the dataset at data/reviews.csv.")

    C.section("2 · Run Multi-Agent Analysis", "Agent 1 → Agent 2 (critique loop, max 2 rounds) → Agent 3, for every selected review")
    with st.container(border=True):
        a, b, d = st.columns([2, 2, 1])
        max_n = len(c.df) if c.df is not None else 1
        n = a.number_input("Number of reviews to analyse", 1, max(max_n, 1), min(10 if c.llm.available else 100, max_n),
                           step=10, key="run_n", help="LLM mode makes about 3–6 API calls per review.")
        how = b.radio("Selection", ["Random sample", "First reviews in the file"], key="run_how", horizontal=True)
        seed = d.number_input("Seed", 0, 9999, 42, key="run_seed", help="Same seed = same random sample.")
        if st.button("▶ Run Multi-Agent Analysis", type="primary", disabled=c.df is None, key="run_go", width="stretch"):
            sub = c.df.sample(int(n), random_state=int(seed)) if how == "Random sample" else c.df.head(int(n))
            states = run_batch_with_progress([row_to_review(r) for _, r in sub.iterrows()], "Analysing")
            remember(states, "run")
            st.session_state["last_run"] = _status_counts(states)
            if states:
                st.session_state["current_key"] = f"run:{states[-1].review.review_id}"
        with st.expander("Analyse a single review instead (typed text, example or test scenario)"):
            _single_review_input(c)

    C.section("3 · Execution status", "how each review was really processed")
    last = st.session_state.get("last_run")
    total = _status_counts(analysed() + [s for s in store().values() if s.agent3 is None])
    if last:
        st.caption("Last run: " + " · ".join(f"{k}: {v}" for k, v in last.items()))
    C.kpi_row([("Analysed successfully", sum(v for k, v in total.items() if k != "Failed"), "all runs in this session", "navy"),
               ("LLM analysis", total["LLM"], "every LLM step succeeded"),
               ("Rule-based fallback", total["Rule-based fallback"] + total["Rule-based (offline)"],
                f"{total['Rule-based fallback']} after an LLM failure · {total['Rule-based (offline)']} offline",
                "warn" if total["Rule-based fallback"] else ""),
               ("Failed analysis", total["Failed"], "workflow did not complete", "crit" if total["Failed"] else "")])
    fatal = st.session_state.get("llm_fatal")
    if fatal:
        st.error("LLM unavailable: " + fatal, icon="⛔")
    if store():
        a, b = st.columns([3, 1])
        a.caption("Results stay available on every page for this session.")
        if b.button("Clear all results", key="clear_all", width="stretch"):
            st.session_state["store"] = {}
            st.session_state.pop("last_run", None)
            st.session_state.pop("ins_pdf", None)
            st.rerun()
        _goto("agent1", "Next: Agent 1 — Feedback Analysis")


# =========================================================================== #
# 5. Executive Dashboard (management view, includes Service Quality Insights)
# =========================================================================== #
def page_executive() -> None:
    import insights as INS
    import report_pdf

    c = ctx()
    C.hero(config.APP_TITLE_EN, "Executive view of beneficiary feedback and where to act first.", [PROTOTYPE],
           title_ar=config.APP_TITLE_AR)
    states = analysed()
    total = c.info["rows_available"] if c.info else None
    ins = INS.compute_insights(states, dataset_reviews=total)
    ins["sla_targets"] = c.sla_targets
    cov = ins["coverage"]

    C.section("Full dataset", "all reviews in the loaded file")
    if c.info:
        C.kpi_row([("Total reviews", f"{c.info['rows_in_file']:,}", "rows in the file", "navy"),
                   ("Unique reviews", f"{c.info['rows_available']:,}", f"{c.info['duplicates_dropped']} duplicates removed", "navy"),
                   ("Applications", c.info["apps"], "government apps", "navy"),
                   ("Average rating", f"{c.df['rating'].mean():.2f} ★", "full dataset", "navy")])
    C.section("Analysed reviews", "results of the three agents — only reviews actually analysed")
    if not states:
        C.empty_state("📊", "No analysis yet — no findings are shown", "Run the analysis on the Data & Run Analysis page.")
        _goto("data", "Go to Data & Run Analysis")
        return
    urgent = sum(n for p, n, _ in ins["priority"] if p in INS.URGENT)
    C.kpi_row([("Reviews analysed", f"{cov['reviews']:,}", f"coverage {cov['coverage_pct']}% of unique reviews" if cov["coverage_pct"] is not None else "—"),
               ("Service issues", cov["issues"], "evidence-backed"),
               ("Actionable feedback", ins["actionable"], f"{ins['actionable'] / cov['reviews'] * 100:.0f}% of analysed reviews"),
               ("P1 / P2 issues", urgent, "attention first", "warn" if urgent else "")])
    if not ins["sample_ok"]:
        st.warning(f"Small sample ({cov['reviews']} review(s)): the findings describe this sample only. Management recommendations "
                   f"are generated from {ins['min_reviews']} analysed reviews or more.")

    with st.container(border=True):
        a, b = st.columns([3, 2])
        a.markdown("**Executive report (PDF, 2–4 pages)** · bilingual headings, built only from the results on this page.")
        if b.button("📄 Generate Executive Report", type="primary", width="stretch", key="ins_gen"):
            with st.spinner("Building the PDF…"):
                st.session_state["ins_pdf"] = (report_pdf.build_executive_report(ins), cov["reviews"], cov["analysis_date"])
        pdf = st.session_state.get("ins_pdf")
        if pdf:
            data, n, day = pdf
            if n != cov["reviews"]:
                b.caption(f"This PDF covers {n} reviews; {cov['reviews']} are analysed now — generate it again.")
            b.download_button("⬇️ Download the PDF", data, f"BeneficiaryVoice_executive_report_{day}.pdf", "application/pdf",
                              width="stretch", key="ins_dl")

    g1, g2 = st.columns(2)
    with g1, st.container(border=True):
        st.markdown("**Most affected service-quality dimensions** (provisional EGSQUAL)")
        if ins["dimensions"]:
            st.altair_chart(CH.single_hue_bars(pd.Series([d["name"].split(" (")[0] for d in ins["dimensions"] for _ in range(d["issues"])]),
                                               T.NAVY, "Issues"), width="stretch")
        else:
            st.caption("No operational issue extracted.")
    with g2, st.container(border=True):
        st.markdown("**Priority distribution** · Impact × Urgency matrix")
        pr = pd.Series([p for p, n, _ in ins["priority"] for _ in range(n)])
        if len(pr):
            st.altair_chart(CH.priority_bars(pr), width="stretch")
        else:
            st.caption("No operational issue extracted.")

    C.section("Most frequently reported issues", "verbatim beneficiary evidence")
    if ins["top_issues"]:
        _rtl_table(pd.DataFrame([{"Issue": t["issue"], "Dimension": t["dimension"].split(" (")[0], "Issues": t["count"],
                                  "P1/P2": t["urgent"], "Example evidence": t["examples"][0]["evidence"] if t["examples"] else ""}
                                 for t in ins["top_issues"][:5]]), ("Example evidence",))
    C.section("Improvement opportunities", "from Agent 3's decisions · recommendations state their basis")
    a, b = st.columns([2, 3])
    with a, st.container(border=True):
        st.markdown("**Actionable feedback** (Agent 3 action per review)")
        st.dataframe(pd.DataFrame(ins["actions"], columns=["action", "reviews", "share %"]), hide_index=True, width="stretch")
    with b, st.container(border=True):
        st.markdown("**Management recommendations**")
        if ins["recommendations"]:
            for k, r in enumerate(ins["recommendations"], start=1):
                st.markdown(f"**{k}. {r['title']}**  \n{r['basis']}")
        else:
            st.caption(f"Not generated: at least {ins['min_reviews']} analysed reviews are needed (currently {cov['reviews']}).")
    with st.expander("More service-quality insights: applications, high-priority feedback, dimensions"):
        st.dataframe(pd.DataFrame([{"application": x["app"], "reviews": x["reviews"], "issues": x["issues"], "P1/P2": x["urgent"],
                                    "most affected dimension": x["top_dimension"], "most frequent issue": x["top_issue"]}
                                   for x in ins["by_app"]]), hide_index=True, width="stretch")
        if ins["high_priority"]:
            _rtl_table(pd.DataFrame([{"Priority": r["priority"], "Application": r["app"], "Issue": r["description"],
                                      "Evidence": r["evidence"]} for r in ins["high_priority"][:10]]), ("Evidence",))
        st.dataframe(pd.DataFrame([{"dimension": d["name"], "EGSQUAL (provisional)": d["egsqual"] or "—", "issues": d["issues"],
                                    "share %": d["share"], "High/Critical": d["high_critical"], "P1/P2": d["urgent"]}
                                   for d in ins["dimensions"]]), hide_index=True, width="stretch")
    st.caption("🛡️ Tickets, SLA targets and escalations are simulated; rules are research-defined. Nothing is sent to any government system.")


# =========================================================================== #
# Settings (secondary)
# =========================================================================== #
def page_settings() -> None:
    from llm import health_check

    c = ctx()
    C.page_header("⚙️ Settings", "Reasoning engine, review rounds, simulated SLA targets and the Impact × Urgency matrix.")
    a, b = st.columns(2)
    with a, st.container(border=True):
        st.markdown("**Reasoning engine**")
        providers = ["anthropic", "openai", "offline"]
        prov = st.selectbox("AI model provider", providers, index=providers.index(st.session_state.get("cfg_provider", "anthropic")))
        st.session_state["cfg_provider"] = prov
        st.caption(("🧠 " + c.llm.name) if c.llm.available else ("⚙️ Offline rule-based mode. " + (c.llm.note or "")))
        if not presentation():
            s = config.llm_settings_summary()
            st.caption(f"Anthropic model: {s['anthropic_model']} · key {'set' if s['anthropic_key_set'] else 'not set'} · "
                       f"OpenAI model: {s['openai_model']} · key {'set' if s['openai_key_set'] else 'not set'} (keys are never shown)")
        if st.button("Test LLM connection", disabled=not c.llm.available,
                     help="Sends one tiny request to check the key, model, credit and structured output."):
            r = health_check(c.llm)
            st.session_state["llm_fatal"] = "" if r["ok"] else r["message"]
            (st.success if r["ok"] else st.error)(r["message"])
        if st.session_state.get("llm_fatal"):
            st.error("Current status: " + st.session_state["llm_fatal"])
    with b, st.container(border=True):
        st.markdown("**Workflow**")
        st.session_state["cfg_rev"] = st.slider("Max Agent 1 ↔ Agent 2 critique rounds", 0, 2, st.session_state.get("cfg_rev", config.MAX_REVISIONS))
        st.session_state["cfg_date"] = st.date_input("SLA reference date (complaint-SLA check)", st.session_state.get("cfg_date", c.ref_date))
        st.session_state["cfg_presentation"] = st.toggle("Management Presentation Mode", st.session_state.get("cfg_presentation", False),
                                                         help="Hides developer diagnostics; keeps fallback, simulation and limitation labels.")
    a, b = st.columns(2)
    with a, st.container(border=True):
        st.markdown("**Simulated SLA targets** (research-defined, elapsed clock hours)")
        sla = st.data_editor(pd.DataFrame([{"priority": p, "response_h": v["response_hours"], "resolution_h": v["resolution_hours"]}
                                           for p, v in c.sla_targets.items()]), key="set_sla", hide_index=True, disabled=["priority"],
                             width="stretch", column_config={"response_h": st.column_config.NumberColumn(min_value=1, step=1),
                                                             "resolution_h": st.column_config.NumberColumn(min_value=1, step=1)})
        st.session_state["cfg_sla"] = {r.priority: {"response_hours": int(r.response_h), "resolution_hours": int(max(r.resolution_h, r.response_h))}
                                       for r in sla.itertuples()}
    with b, st.container(border=True):
        st.markdown("**Impact × Urgency matrix** (rows = impact, columns = urgency)")
        lv = ["High", "Medium", "Low"]
        m = st.data_editor(pd.DataFrame([{"impact": i, **c.matrix[i]} for i in lv]), key="set_matrix", hide_index=True,
                           disabled=["impact"], width="stretch",
                           column_config={u: st.column_config.SelectboxColumn(u, options=["P1", "P2", "P3", "P4"], required=True) for u in lv})
        st.session_state["cfg_matrix"] = {r.impact: {u: getattr(r, u) for u in lv} for r in m.itertuples()}
    if st.button("Reset SLA targets and matrix to the documented defaults"):
        for k in ("cfg_sla", "cfg_matrix", "set_sla", "set_matrix"):
            st.session_state.pop(k, None)
        st.rerun()
    st.caption("Changes apply to reviews analysed from now on; earlier results keep the settings they were analysed with.")


# =========================================================================== #
# Methodology & References (secondary)
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

    C.page_header("📘 Methodology & References", "Published frameworks inform the design; every scoring rule, threshold and "
                  "SLA value is research-defined and editable in knowledge_base/.")
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
        st.dataframe(pd.DataFrame(mk["impact_rules"] + mk["urgency_rules"] + [{"id": mk["verification_rule"]["id"], "level": "cap",
                                                                              "rule": mk["verification_rule"]["rule"]}]),
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
