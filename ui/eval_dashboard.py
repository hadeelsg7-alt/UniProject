"""Evaluation Dashboard: measures the evaluation plan's metrics against
human-verified reference labels (Agent 1, Agent 2, Agent 3, overall framework).

Nothing is estimated or simulated: without uploaded labels no metric is shown,
and baselines that cannot run (no LLM) are reported as not run."""
from __future__ import annotations

import hashlib
import json

import pandas as pd
import streamlit as st

import baselines
import evaluation_suite as E
from ui import charts as CH
from workflow import MultiAgentWorkflow
from ui import components as C

A_COLORS = {k: C.AGENTS[k]["color"] for k in ("Agent1", "Agent2", "Agent3")}
FIELD_TITLES = {"feedback_type": "Feedback type", "dimensions": "Service domain (EGSQUAL)", "sentiment": "Sentiment",
                "severity": "Severity", "priority": "Priority"}


def _f(v, fmt: str = ".3f", na: str = "n/a") -> str:
    return na if v is None or (isinstance(v, float) and pd.isna(v)) else format(v, fmt)


def _ctx():
    return st.session_state["ctx"]


# --------------------------------------------------------------------------- #
def page_eval_dashboard(show_hero: bool = True) -> None:
    c = _ctx()
    if show_hero:
        C.page_header("📈 Evaluation & Results", "How well the three agents perform: session results and collaboration, then "
                      "accuracy against human-verified labels and the baseline comparison. No score is ever estimated.")
    _session_results(c)
    C.section("Ground-truth evaluation", "metrics appear only after human-verified labels are uploaded")
    labels, ref = _step_labels(c)
    if labels is None:
        return
    run = _step_run(c, ref, labels)
    if run is None:
        return
    _results(run, labels)


def _session_results(c) -> None:
    """Label-free view of the session's analysis results: collaboration, decision consistency, exports."""
    from ui.pages import analysed

    states = analysed()
    if not states:
        st.caption("No analysed reviews in this session yet — run the analysis on the Data & Run Analysis page.")
        return
    with st.expander(f"Session results and agent collaboration · {len(states)} review(s) analysed (no labels needed)"):
        _session_body(states)


def _session_body(states) -> None:
    from workflow import state_to_dict, state_to_row
    preds = [E.predictions_from_state(s) for s in states]
    col = E.collaboration_summary(preds)
    C.kpi_row([("Reviews analysed", col["reviews"], "three-agent workflow"),
               ("Reviews with a critique", col["with_review_request"], f"{col['review_requests']} request(s) Agent 2 → Agent 1"),
               ("Revision rounds", col["revision_rounds"], f"max {col['max_rounds_used']} per review (limit 2)"),
               ("Rule-based fallback", col["llm_fallback_reviews"], "reviews with an LLM failure", "warn" if col["llm_fallback_reviews"] else "")])
    st.markdown("**Evidence and decision consistency** (label-free checks)")
    st.dataframe(E.consistency_rates(preds), hide_index=True, width="stretch")
    st.caption("These checks test the agents against their own documented rules and the review text; they are not accuracy.")
    table = pd.DataFrame([state_to_row(s) for s in states])
    a, b = st.columns(2)
    a.download_button("⬇️ Results (CSV)", table.to_csv(index=False).encode("utf-8-sig"), "analysis_results.csv", "text/csv",
                      width="stretch")
    b.download_button("⬇️ Full results with collaboration traces (JSON)",
                      json.dumps([state_to_dict(s) for s in states], ensure_ascii=False, indent=1).encode("utf-8"),
                      "analysis_results.json", "application/json", width="stretch")


# --------------------------------------------------------------------------- #
# Step 1: reference labels
# --------------------------------------------------------------------------- #
def _step_labels(c):
    C.section("1 · Upload the reference labels", "human-verified labels, e.g. evaluation_labels/review_labels_50.xlsx")
    with st.expander("Optional: create a new annotation sheet, or see the allowed label values"):
        a, b = st.columns([3, 2])
        with a, st.container(border=True):
            st.markdown("**Create an annotation sheet** · stratified random sample by star rating (seed 42). "
                        "Label columns are left **blank** on purpose — the system's predictions are not shown to annotators.")
            if c.df is None:
                st.caption("Load the reviews dataset (sidebar) to export a sample, or upload your own labelled file with a `text` column.")
            else:
                n = st.slider("Sample size", 20, min(500, len(c.df)), min(100, len(c.df)), step=10, key="evd_n")
                sheet = E.annotation_sheet(c.df, n=n)
                d1, d2 = st.columns(2)
                d1.download_button("⬇️ Annotation sheet (CSV)", sheet.to_csv(index=False).encode("utf-8-sig"),
                                   f"annotation_sheet_{n}.csv", "text/csv", width="stretch")
                d2.download_button("⬇️ Annotation guidelines", E.annotation_guidelines().encode("utf-8"),
                                   "annotation_guidelines.md", "text/markdown", width="stretch")
        with b, st.container(border=True):
            st.markdown("**Label columns** (any subset; empty cell = not labelled)")
            st.dataframe(pd.DataFrame([{"column": k, "agent": C.agent_name(v["agent"]), "values": ", ".join(v["values"] or (E.team_names() if k == "handling_team" else ["EGSQUAL keys ';'"]))}
                                       for k, v in E.LABEL_SPEC.items()]), hide_index=True, width="stretch", height=320)

    up = st.file_uploader("Upload the labels: CSV, or the Excel review workbook (must contain `review_id`; optional `text`, `rating`, `app_id`)",
                          type=["csv", "xlsx"], key="evd_labels")
    if up is None:
        C.empty_state("🔒", "No reference labels → no metrics", "Precision, Recall, F1 and the system comparison are computed "
                      "only from human-verified labels. Nothing is estimated or simulated without them.")
        return None, None
    try:
        ref, original = E.read_labels_file(up.getvalue(), up.name)
        labels, cov = E.normalise_reference(ref)
    except ImportError:
        st.error("Reading Excel needs the openpyxl package: run `pip install openpyxl` in the project's .venv, or upload a CSV.")
        return None, None
    except Exception as e:  # noqa: BLE001
        st.error(f"Could not read the labels: {e}")
        return None, None
    st.session_state["evd_fp"] = hashlib.sha1(up.getvalue()).hexdigest()[:12]
    rs = E.review_status(ref)
    if rs["tracked"] and rs["unreviewed"]:
        st.warning(f"⚠️ {rs['unreviewed']} of {rs['rows']} row(s) are **not yet marked `{E.VERIFIED}`**. Those labels were "
                   "drafted by an AI (Claude). The framework also uses Claude, so scores against unreviewed labels are biased "
                   "and must not be reported as human-verified results.")
    elif rs["tracked"]:
        st.success(f"✅ All {rs['rows']} row(s) are marked `{E.VERIFIED}` by the reviewer.")
    if original is not None:
        ch = E.label_changes(ref, original)
        tot_d, tot_c = int(ch["drafted"].sum()), int(ch["changed_by_reviewer"].sum())
        with st.expander(f"Human review of the AI draft · {tot_c} of {tot_d} label(s) changed "
                         f"({100 * tot_c / max(tot_d, 1):.1f}%)"):
            st.dataframe(ch, hide_index=True, width="stretch")
            st.caption("Report this in the methodology: labels were pre-annotated by an LLM and reviewed/corrected by the researcher.")
    with st.expander(f"Label coverage · {len(labels)} labelled review(s)", expanded=bool(cov['unrecognised'].sum())):
        st.dataframe(cov, hide_index=True, width="stretch")
        if cov["unrecognised"].sum():
            st.warning("Some cells were not recognised and are excluded (never guessed). Fix them using the allowed values.")
    if not labels:
        C.empty_state("⚠️", "No usable labels", "The file has a review_id column but no recognised label values.")
        return None, None
    return labels, ref


# --------------------------------------------------------------------------- #
# Step 2: run the systems on the test set
# --------------------------------------------------------------------------- #
def _step_run(c, ref: pd.DataFrame, labels: dict):
    C.section("2 · Run the systems on the same test set", "timed per review; tokens counted from the API responses")
    reviews, missing = E.reviews_for_labels(ref[ref["review_id"].astype(str).str.strip().isin(labels)], c.df)
    if missing:
        st.info(f"{len(missing)} labelled review_id(s) have no text (not in the loaded dataset and no `text` column): "
                + ", ".join(missing[:10]) + ("…" if len(missing) > 10 else ""))
    if not reviews:
        C.empty_state("⚠️", "Nothing to run", "None of the labelled reviews could be resolved to a text.")
        return None
    llm_ok = c.llm.available
    options = list(E.SYSTEMS)
    plan = ["multi_agent", "single_llm", "sequential"]  # the three systems of the evaluation plan
    sel = st.multiselect("Systems (the evaluation plan compares the first three; the other two are optional extras)", options,
                         default=plan if llm_ok else ["multi_agent"],
                         format_func=E.SYSTEMS.get, key="evd_systems")
    with st.expander("What each system is (same Claude model and same knowledge for all — only the architecture differs)"):
        st.dataframe(pd.DataFrame([{"system": E.SYSTEMS[k], "role": E.SYSTEM_ROLE[k]} for k in E.SYSTEMS]),
                     hide_index=True, width="stretch")
    if not llm_ok:
        st.warning("Offline mode: the LLM baselines (Single LLM, few-shot + CoT, Sequential) need a working LLM, so they will be "
                   "recorded as **not run** (failed), never simulated. The multi-agent framework and its ablation run with "
                   "the rule-based reasoner.")
    calls = {"multi_agent": "≈3–6", "single_llm": "1", "single_llm_cot": "1", "sequential": "3", "multi_agent_no_loop": "≈3–4"}
    st.caption(f"{len(reviews)} review(s) × {len(sel)} system(s). LLM calls per review: "
               + " · ".join(f"{E.SYSTEMS[s]} {calls[s]}" for s in sel) + (" (offline: 0)" if not llm_ok else ""))

    prev = st.session_state.get("evd_run_result")
    keep = (prev is not None and prev.engine == c.llm.name and st.session_state.get("evd_run_fp") == st.session_state.get("evd_fp"))
    if keep:
        st.caption("Already run: " + ", ".join(E.SYSTEMS[s] for s in prev.results)
                   + ". Running again replaces only the selected systems, so you can add a system without paying for the others again.")
    if st.button("▶ Run evaluation", type="primary", key="evd_run", disabled=not sel):
        run = prev if keep else E.EvalRun(engine=c.llm.name)
        for s in sel:
            bar = st.progress(0.0, text=f"{E.SYSTEMS[s]}: 0/{len(reviews)}")
            prog = lambda k, n, s=s, bar=bar: bar.progress(k / n, text=f"{E.SYSTEMS[s]}: {k}/{n}")  # noqa: E731
            if s == "multi_agent":
                run.results[s] = E.run_multi_agent(c.workflow, reviews, reference_date=c.ref_date, progress=prog)
            elif s == "multi_agent_no_loop":
                wf = MultiAgentWorkflow(c.llm, max_revisions=0)
                run.results[s] = E.run_multi_agent(wf, reviews, reference_date=c.ref_date, progress=prog, system=s)
            else:
                run.results[s] = baselines.run_baseline(s, c.llm, reviews, progress=prog)
            run.prices[s] = c.llm.price_per_mtok()
            bar.empty()
        run.results = {k: run.results[k] for k in E.SYSTEMS if k in run.results}  # stable order
        st.session_state["evd_run_result"] = run
        st.session_state["evd_reviews"] = {r.review_id: r for r in reviews}
        st.session_state["evd_run_fp"] = st.session_state.get("evd_fp")
    run = st.session_state.get("evd_run_result")
    if run is None:
        C.empty_state("🧪", "Not run yet", "Choose the systems and click Run evaluation.")
        return None
    if st.session_state.get("evd_run_fp") != st.session_state.get("evd_fp"):
        st.info("The labels file changed since the last run. Metrics below use the new labels on the previous predictions; "
                "reviews without a prediction are ignored. Re-run to include new reviews.")
    return run


# --------------------------------------------------------------------------- #
# Results
# --------------------------------------------------------------------------- #
# Headline metrics: one line per agent, the numbers the research reports first
HEADLINE = {
    "Agent1": [("Sentiment", "sentiment", "macro_f1", "Macro F1"), ("Feedback type", "feedback_type", "macro_f1", "Macro F1"),
               ("EGSQUAL dimension", "dimensions", "macro_f1", "Macro F1")],
    "Agent2": [("Severity", "severity", "accuracy", "Accuracy"), ("Priority", "priority", "accuracy", "Accuracy"),
               ("Priority error", "priority", "mae", "MAE (lower is better)")],
    "Agent3": [("Actionability", "ticket", "accuracy", "Accuracy"), ("Routing", "handling_team", "accuracy", "Accuracy"),
               ("Escalation", "escalation", "accuracy", "Accuracy")],
}
AGENT_TITLE = {"Agent1": "Agent 1 — Feedback Analysis", "Agent2": "Agent 2 — Impact & Priority",
               "Agent3": "Agent 3 — Resolution & Escalation"}


def _results(run: E.EvalRun, labels: dict) -> None:
    sys_keys = list(run.results)
    main = "multi_agent" if "multi_agent" in sys_keys else sys_keys[0]
    preds = run.results[main]
    C.section("3 · Results", f"{len(labels)} labelled review(s) · engine: {run.engine}")
    failed = sum(p.failed for p in preds)
    if failed:
        st.error(f"⛔ {failed} of {len(preds)} review(s) produced no output — counted as wrong, not removed.")
    fb = sum(p.used_fallback for p in preds)
    if fb:
        st.warning(f"{fb} review(s) used the rule-based fallback (no LLM credit or an LLM error).")
    _glance(run, labels, main)
    _versus(run, labels)
    st.write("")
    st.markdown("##### Details")
    t0, t1, t2, t3, t4, t5 = st.tabs(["📋 Summary table (3.5)", "Agent 1", "Agent 2", "Agent 3", "Time & cost", "Statistical tests"])
    with t0:
        _plan_summary(run, labels)
    show = main
    if len(sys_keys) > 1:
        with t1:
            show = st.segmented_control("Agent details for", sys_keys, default=main, format_func=E.SYSTEMS.get,
                                        key="evd_show") or main
    dp = run.results[show]
    with t1:
        _agent1(dp, labels)
    with t2:
        _agent2(dp, labels)
    with t3:
        _agent3(dp, labels)
    with t4:
        _overall(run, labels)
    with t5:
        _proof(run, labels)


def _glance(run: E.EvalRun, labels: dict, main: str) -> None:
    """Three cards, one per agent, with the three numbers that matter most."""
    preds = run.results[main]
    cols = st.columns(3)
    for col, (ag, items) in zip(cols, HEADLINE.items()):
        color = A_COLORS[ag]
        rows = ""
        for name, f, metric, sub in items:
            v = E.plan_metric(labels, preds, f, metric)
            val = _f(v, ".2f" if metric == "mae" else ".0%" if metric == "accuracy" else ".2f")
            rows += (f'<div style="display:flex;justify-content:space-between;align-items:baseline;padding:8px 0;'
                     f'border-top:1px solid #EDF3EF"><div><div style="font-weight:600">{name}</div>'
                     f'<div class="small">{sub}</div></div>'
                     f'<div style="font-size:1.7rem;font-weight:800;color:{color if v is not None else "#9AA8A1"}">{val}</div></div>')
        with col:
            C.md(f'<div class="kpi" style="border-top:6px solid {color};padding:14px 16px">'
                 f'<div style="font-weight:700;color:{color};margin-bottom:4px">{AGENT_TITLE[ag]}</div>{rows}</div>')
    st.caption(f"{E.SYSTEMS[main]} · n/a = no reference labels for that field (nothing is estimated).")


VERSUS = [("Sentiment", "sentiment", "macro_f1"), ("Feedback type", "feedback_type", "macro_f1"),
          ("EGSQUAL dimension", "dimensions", "macro_f1"), ("Severity", "severity", "accuracy"),
          ("Priority", "priority", "accuracy"), ("Actionability", "ticket", "accuracy"), ("Escalation", "escalation", "accuracy")]


def _versus(run: E.EvalRun, labels: dict) -> None:
    """Proposed framework vs. baselines on the same headline metrics, with a one-line verdict."""
    with st.container(border=True):
        st.markdown("**Multi-agent framework vs. baselines** (higher is better)")
        if len(run.results) < 2 or "multi_agent" not in run.results:
            st.caption("Only one system was run. Select Single LLM and Sequential in step 2 (needs a working LLM) to compare.")
            return
        rows = []
        for name, f, m in VERSUS:
            for sk, preds in run.results.items():
                v = E.plan_metric(labels, preds, f, m)
                if v is not None:
                    rows.append({"metric": name, "system_key": sk, "system": E.SYSTEMS[sk], "value": float(v)})
        long = pd.DataFrame(rows)
        if long.empty:
            st.caption("No labelled fields to compare.")
            return
        wins = ties = total = 0
        for name, g in long.groupby("metric", sort=False):
            if "multi_agent" not in set(g["system_key"]) or len(g) < 2:
                continue
            total += 1
            ma = g[g["system_key"] == "multi_agent"]["value"].iloc[0]
            best_other = g[g["system_key"] != "multi_agent"]["value"].max()
            wins += ma > best_other + 1e-9
            ties += abs(ma - best_other) <= 1e-9
        if total:
            st.markdown(f"The multi-agent framework is **best on {wins} of {total}** metric(s)"
                        + (f", tied on {ties}" if ties else "") + f", and behind on {total - wins - ties}. "
                        "See *Statistical tests* for whether the differences are significant.")
        wide = long.pivot_table(index="metric", columns="system", values="value", sort=False)
        wide = wide[[E.SYSTEMS[k] for k in run.results if E.SYSTEMS[k] in wide.columns]]
        wide.index.name = None
        st.dataframe(wide.style.format("{:.2f}", na_rep="n/a")
                     .highlight_max(axis=1, props="font-weight:700;background-color:#DDF3E8;color:#0B3D2E"),
                     width="stretch")
        st.caption("Green cell = best system for that metric. Sentiment, feedback type and EGSQUAL use Macro F1; the others use accuracy.")


def _plan_summary(run: E.EvalRun, labels: dict) -> None:
    """Section 3.5 of the research: every planned metric for every system, in one table."""
    tab = E.evaluation_plan_table(labels, run.results)
    with st.container(border=True):
        st.markdown("**Evaluation summary (research method, Section 3.5)** — Agent 1: Precision, Recall, Macro F1 · "
                    "Agent 2: Accuracy, Macro F1, MAE · Agent 3: Accuracy, F1, consistency · compared across systems")
        st.dataframe(tab, hide_index=True, width="stretch")
        st.caption("Higher is better for every metric except MAE (lower is better). Empty cells mean the metric could not be "
                   "computed: no reference labels for that field, or the system does not produce it (baselines do not route "
                   "to teams or run rule-consistency checks). Nothing is estimated.")
        st.download_button("⬇ Download table (CSV)", tab.to_csv(index=False).encode("utf-8-sig"),
                           file_name="evaluation_summary_section_3_5.csv", mime="text/csv", key="evd_plan_csv")


def _multiclass_block(title: str, m: dict, color: str, order: list[str] | None, key: str) -> None:
    with st.container(border=True):
        st.markdown(f"**{title}** · {m['n']} labelled review(s)")
        if not m["n"]:
            st.caption("No reference labels for this field.")
            return
        C.kpi_row([("Macro F1", _f(m["macro_f1"]), "mean over classes"), ("Precision", _f(m["macro_precision"]), "macro"),
                   ("Recall", _f(m["macro_recall"]), "macro"), ("Accuracy", _f(m["accuracy"]), "exact match", "navy")])
        t1, t2, t3 = st.tabs(["Per class", "Confusion matrix", "Table"])
        with t1:
            if not m["table"].empty:
                st.altair_chart(CH.prf_bars(m["table"]), width="stretch", key=f"{key}_prf")
        with t2:
            st.altair_chart(CH.confusion_heatmap(m["confusion"], color, order), width="stretch", key=f"{key}_cm")
        with t3:
            st.dataframe(m["table"], hide_index=True, width="stretch")


def _agent1(preds, labels) -> None:
    m = E.agent1_metrics(labels, preds)
    with st.container(key="frame_Agent1_evd"):
        C.agent_section_header("Agent1", "Agent 1 evaluation: sentiment, feedback classification, EGSQUAL dimensions, issue extraction "
                                         "· Precision, Recall, Macro F1")
        _multiclass_block("Sentiment identification", m["sentiment"], A_COLORS["Agent1"], E.SENTIMENTS, "evd_sent")
        _multiclass_block("Feedback classification", m["feedback_type"], A_COLORS["Agent1"], E.FEEDBACK_TYPES, "evd_ft")
        d = m["dimensions"]
        with st.container(border=True):
            st.markdown(f"**Service domain (EGSQUAL dimensions, multi-label)** · {d['n']} labelled review(s)")
            if not d["n"]:
                st.caption("No reference labels for this field.")
            else:
                C.kpi_row([("Macro F1", _f(d["macro_f1"]), "mean over dimensions"), ("Precision", _f(d["macro_precision"]), "macro"),
                           ("Recall", _f(d["macro_recall"]), "macro"), ("Micro F1", _f(d["micro_f1"]), "all review×dimension pairs", "navy"),
                           ("Exact set match", _f(d["exact_match"]), "", "navy")])
                if not d["table"].empty:
                    st.altair_chart(CH.prf_bars(d["table"]), width="stretch", key="evd_dim_prf")
                    st.dataframe(d["table"], hide_index=True, width="stretch")
        ic = m["issue_count"]
        with st.container(border=True):
            st.markdown(f"**Issue extraction** (number of independent issues per review) · {ic['n']} labelled review(s)")
            if not ic["n"]:
                st.caption("Add an `issue_count` column to the labels to evaluate issue extraction.")
            else:
                C.kpi_row([("Exact issue-count accuracy", _f(ic["accuracy"]), "reviews with the right number of issues"),
                           ("Macro F1", _f(ic["macro_f1"]), "over issue-count classes")])


def _agent2(preds, labels) -> None:
    m = E.agent2_metrics(labels, preds, st.session_state.get("evd_reviews"))
    with st.container(key="frame_Agent2_evd"):
        C.agent_section_header("Agent2", "Agent 2 evaluation: severity and priority · Accuracy, Macro F1, MAE")
        with st.container(border=True):
            st.markdown("**Star rating vs. Agent 1 sentiment** (1–2★ negative · 3★ neutral · 4–5★ positive) — context only")
            rb = m["rating_baseline"]
            C.kpi_row([("Agreement: predicted sentiment vs. rating", _f(m["rating_agreement"], ".1%"),
                        f"{m['rating_agreement_n']} review(s) with a rating", "navy"),
                       ("Rating-only Macro F1 vs. labels", _f(rb["macro_f1"]) if rb else "n/a",
                        "reference point: sentiment from stars alone", "navy"),
                       ("System Macro F1 vs. labels", _f(m["sentiment"]["macro_f1"]), "from the table above")])
            st.caption("The rating is never used as a label and never changes severity; it only helps interpret sentiment.")
        _multiclass_block("Severity", m["severity"], A_COLORS["Agent2"], E.SEVERITIES, "evd_sev")
        _multiclass_block("Priority", m["priority"], A_COLORS["Agent2"], E.PRIORITIES, "evd_pri")
        with st.container(border=True):
            st.markdown("**Mean Absolute Error on the ordinal scale** (None 0 · Low/P4 1 · Medium/P3 2 · High/P2 3 · Critical/P1 4)")
            sm, pm = m["severity_mae"], m["priority_mae"]
            C.kpi_row([("Severity MAE", _f(sm["mae"], ".2f"), f"{sm['n']} review(s) · lower is better"),
                       ("Severity within ±1 level", _f(sm["within_one"], ".1%"), f"max error {sm['max_error'] if sm['max_error'] is not None else 'n/a'}", "navy"),
                       ("Priority MAE", _f(pm["mae"], ".2f"), f"{pm['n']} review(s) · lower is better"),
                       ("Priority within ±1 level", _f(pm["within_one"], ".1%"), f"max error {pm['max_error'] if pm['max_error'] is not None else 'n/a'}", "navy")])
            ex = sm["excluded_failed"] + pm["excluded_failed"]
            st.caption("MAE measures how far a wrong level is from the reference: predicting P2 for a P1 case (error 1) is "
                       "less serious than predicting P4 (error 3)." + (f" {ex} failed prediction(s) have no level and are "
                       "excluded from MAE (they already count as wrong in accuracy and F1)." if ex else ""))


def _binary_kpis(title: str, b: dict) -> None:
    with st.container(border=True):
        st.markdown(f"**{title}** · {b['n']} labelled review(s)")
        if not b["n"]:
            st.caption("No reference labels for this field.")
            return
        C.kpi_row([("F1", _f(b["f1"]), "positive class 'yes'"), ("Accuracy", _f(b["accuracy"]), "", "navy")])
        C.kpi_row([("Precision", _f(b["precision"]), "of predicted 'yes'"), ("Recall", _f(b["recall"]), "of reference 'yes'")])
        st.caption(f"TP {b['tp']} · FP {b['fp']} · FN {b['fn']} · TN {b['tn']}"
                   + ("" if b["defined"] else " — F1/recall undefined: the labelled set has no reference 'yes' cases."))


def _agent3(preds, labels) -> None:
    m = E.agent3_metrics(labels, preds)
    with st.container(key="frame_Agent3_evd"):
        C.agent_section_header("Agent3", "Agent 3 evaluation: actionability, routing, escalation · Accuracy, F1, rule consistency")
        a, b = st.columns(2)
        with a:
            _binary_kpis("Actionability (ticket generated: yes/no)", m["ticket"])
        with b:
            _binary_kpis("Escalation decision (simulated)", m["escalation"])
        with st.container(border=True):
            st.markdown(f"**JSON ticket validity** · {m['json_total']} generated ticket(s)")
            C.kpi_row([("JSON validity rate", _f(m["json_validity_rate"], ".1%"), f"{m['json_valid']} / {m['json_total']} valid"),
                       ("Checks", "parse · schema · fields", "allowed severity/priority · simulated + human approval", "navy")])
            bad = m["json_checks"][~m["json_checks"]["valid"]] if m["json_total"] else pd.DataFrame()
            if not bad.empty:
                st.dataframe(bad, hide_index=True, width="stretch")
        rt = m["routing"]
        if rt["n"]:
            _multiclass_block("Routing (handling team)", rt, A_COLORS["Agent3"], None, "evd_route")
        else:
            st.caption("Routing accuracy: add a `handling_team` column to the labels (team names as in knowledge_base/handling_teams.json).")
        _multiclass_block("Recommended action (additional)", m["action"], A_COLORS["Agent3"], E.ACTIONS, "evd_act")
        if any(p.consistency for p in preds):
            with st.container(border=True):
                st.markdown("**Consistency of decisions with the identified issues and the predefined rules** (label-free)")
                st.dataframe(E.consistency_rates(preds), hide_index=True, width="stretch")


def _overall(run: E.EvalRun, labels: dict) -> None:
    tab = E.overall_table(labels, run.results, run.prices, st.session_state.get("evd_reviews"))
    colors = E.SYSTEM_COLORS
    with st.container(key="frame_Overall_evd"):
        C.md('<div style="display:flex;align-items:center;gap:12px;margin-bottom:8px">'
             '<span class="agent-tag" style="background:#0E3B2C;font-size:0.8rem;padding:4px 10px">OVERALL</span>'
             '<span class="agent-title" style="font-size:1.15rem">🏛️ Overall framework comparison</span>'
             '<span class="agent-ar" style="font-size:1rem">مقارنة الإطار الكامل</span></div>'
             '<div class="small" style="margin:-2px 0 10px 0">Same test set · Macro F1, Escalation F1, processing time, estimated cost</div>')
        if len(tab) < 2:
            st.info("Only one system was run. Select the baselines (needs a working LLM) to compare architectures.")
        cols = st.columns(len(tab))
        for col, (_, r) in zip(cols, tab.iterrows()):
            with col:
                C.md(f'<div class="kpi" style="border-top:5px solid {colors[r.system_key]}"><div class="label">{r.system}</div>'
                     f'<div class="value">{_f(r.macro_f1_mean)}</div><div class="sub">mean Macro F1 · Escalation F1 {_f(r.escalation_f1)}<br>'
                     f'{_f(r.time_per_review_s, ".2f")} s/review · '
                     f'{"$" + _f(r.est_cost_per_review_usd, ".4f") if r.est_cost_usd is not None else "cost n/a"} /review'
                     f'{" · " + str(int(r.failed_reviews)) + " failed" if r.failed_reviews else ""}</div></div>')
        st.write("")
        long = pd.DataFrame([{"field": FIELD_TITLES[f], "system_key": r.system_key, "system": r.system, "f1": r[f"macro_f1_{f}"]}
                             for _, r in tab.iterrows() for f in E.MACRO_FIELDS if r[f"macro_f1_{f}"] is not None
                             and not pd.isna(r[f"macro_f1_{f}"])])
        with st.container(border=True):
            st.markdown("**Macro F1 per field**")
            if long.empty:
                st.caption("No labelled fields to compare.")
            else:
                st.altair_chart(CH.faceted_system_bars(long, "field", "f1", colors, "Macro F1"), key="evd_ov_f1")
        g1, g2 = st.columns(2)
        with g1, st.container(border=True):
            st.markdown("**Escalation F1**")
            if tab["escalation_f1"].notna().any():
                st.altair_chart(CH.system_metric_bars(tab, "escalation_f1", "F1", colors, domain_max=1.08),
                                width="stretch", key="evd_ov_esc")
            else:
                st.caption("n/a — no reference escalation 'yes' cases in the labelled set.")
        with g2, st.container(border=True):
            st.markdown("**Processing time** (mean seconds per review)")
            st.altair_chart(CH.system_metric_bars(tab, "time_per_review_s", "Seconds", colors, fmt=".2f"),
                            width="stretch", key="evd_ov_time")
        with st.container(border=True):
            st.markdown("**Estimated cost** (USD per review = counted tokens × configured price per million tokens)")
            if tab["est_cost_per_review_usd"].notna().any():
                st.altair_chart(CH.system_metric_bars(tab, "est_cost_per_review_usd", "USD per review", colors, fmt=".4f"),
                                width="stretch", key="evd_ov_cost")
            else:
                st.caption("n/a — offline run (no tokens) or no prices configured for this provider "
                           "(set ANTHROPIC_PRICE_INPUT / _OUTPUT or OPENAI_PRICE_* in .env).")
            st.caption("Estimate only: prices come from configuration, not from your invoice; tokens of calls that failed are not counted.")
        show = tab.drop(columns=["system_key"]).copy()
        st.dataframe(show, hide_index=True, width="stretch")
        d1, d2, d3 = st.columns(3)
        d1.download_button("⬇️ Comparison table (CSV)", tab.to_csv(index=False).encode("utf-8-sig"), "framework_comparison.csv",
                           "text/csv", width="stretch")
        d2.download_button("⬇️ All predictions (CSV)", E.predictions_frame(run.results).to_csv(index=False).encode("utf-8-sig"),
                           "evaluation_predictions.csv", "text/csv", width="stretch")
        d3.download_button("⬇️ Metrics (JSON)", json.dumps(_metrics_json(run, labels), ensure_ascii=False, indent=1, default=str)
                           .encode("utf-8"), "evaluation_metrics.json", "application/json", width="stretch")


def _proof(run: E.EvalRun, labels: dict) -> None:
    """Evidence that the framework is (or is not) better: reliability metrics, significance tests, CIs, hard cases."""
    reviews = st.session_state.get("evd_reviews") or {}
    colors = E.SYSTEM_COLORS
    tab = E.overall_table(labels, run.results, run.prices, reviews)
    with st.container(key="frame_Overall_proof"):
        C.md('<div style="display:flex;align-items:center;gap:12px;margin-bottom:8px">'
             '<span class="agent-tag" style="background:#0E3B2C;font-size:0.8rem;padding:4px 10px">EVIDENCE</span>'
             '<span class="agent-title" style="font-size:1.15rem">🧪 Is the framework better? — statistical evidence</span>'
             '<span class="agent-ar" style="font-size:1rem">هل الإطار أفضل فعلاً؟</span></div>'
             '<div class="small" style="margin:-2px 0 10px 0">Paired tests on the same reviews · 95% bootstrap intervals · '
             'reliability · hard cases. The text below is generated from the numbers, including where the framework is not better.</div>')
        if "multi_agent" not in run.results or len(run.results) < 2:
            st.info("Run the Multi-Agent framework and at least one other system to compare them.")
            return

        sig = E.significance_table(labels, run.results)
        if not sig.empty:
            with st.container(border=True):
                st.markdown("**Summary (alpha = 0.05, exact McNemar test on per-review correctness)**")
                for other, g in sig.groupby("compared_with", sort=False):
                    better = g[g["result (alpha 0.05)"] == "framework better"]["field"].tolist()
                    worse = g[g["result (alpha 0.05)"] == "framework worse"]["field"].tolist()
                    same = g[g["result (alpha 0.05)"] == "not significant"]["field"].tolist()
                    parts = []
                    if better:
                        parts.append("✅ significantly **better** on " + ", ".join(f"`{x}`" for x in better))
                    if worse:
                        parts.append("❌ significantly **worse** on " + ", ".join(f"`{x}`" for x in worse))
                    if same:
                        parts.append("➖ no significant difference on " + ", ".join(f"`{x}`" for x in same))
                    st.markdown(f"- vs **{other}**: " + "; ".join(parts))
                n = int(sig["n"].max())
                if n < 60:
                    st.caption(f"With {n} labelled reviews only large differences can reach significance; "
                               "'no significant difference' means the test set is too small to tell, not that the systems are equal.")
                st.dataframe(sig.style.format({"framework_accuracy": "{:.3f}", "other_accuracy": "{:.3f}", "p_value": "{:.4f}"}),
                             hide_index=True, width="stretch")

        with st.container(border=True):
            st.markdown("**Reliability — what matters for a government service**")
            st.caption("Severe-case recall: of the reviews a human labelled High/Critical, the share the system also flagged High/Critical. "
                       "Evidence grounding: share of quoted evidence that really appears in the review (no invented quotes).")
            a, b = st.columns(2)
            with a:
                if tab["severe_recall"].notna().any():
                    st.altair_chart(CH.system_metric_bars(tab, "severe_recall", "Severe-case recall", colors, fmt=".2f", domain_max=1.08),
                                    width="stretch", key="evd_pf_sev")
                else:
                    st.caption("No High/Critical cases in the labels.")
            with b:
                if tab["evidence_grounding_rate"].notna().any():
                    st.altair_chart(CH.system_metric_bars(tab, "evidence_grounding_rate", "Evidence grounding rate", colors,
                                                          fmt=".2f", domain_max=1.08), width="stretch", key="evd_pf_grd")
                else:
                    st.caption("No evidence quotes produced.")
            st.dataframe(tab[["system", "severe_recall", "evidence_grounding_rate", "reviews_with_unsupported_quote",
                              "json_validity_rate", "failed_reviews"]], hide_index=True, width="stretch")

        with st.container(border=True):
            st.markdown("**Macro F1 with 95% bootstrap confidence intervals** (1,000 resamples of the labelled reviews)")
            ci = E.ci_table(labels, run.results)
            if ci.empty:
                st.caption("Not enough labelled reviews (needs at least 5 per field).")
            else:
                ci["field"] = ci["field"].map(FIELD_TITLES)
                st.altair_chart(CH.ci_chart(ci, colors), key="evd_pf_ci")
                st.caption("Overlapping intervals mean the difference may be due to the particular sample of reviews.")

        with st.container(border=True):
            st.markdown("**Hard cases** — mean accuracy over feedback type, severity and action, per subgroup")
            st.caption("Subgroups are defined from the review text and the human labels only, never from a system's output.")
            sub = E.subgroup_table(labels, run.results, reviews)
            if sub.empty:
                st.caption("No subgroup could be formed.")
            else:
                sub["subgroup"] = sub["subgroup"] + " (n=" + sub["n"].astype(str) + ")"
                st.altair_chart(CH.faceted_system_bars(sub, "subgroup", "mean_accuracy", colors, "Mean accuracy"),
                                key="evd_pf_sub")
                st.dataframe(sub.drop(columns=["system_key"]), hide_index=True, width="stretch")


def _metrics_json(run: E.EvalRun, labels: dict) -> dict:
    def clean(m):
        if isinstance(m, dict):
            return {k: clean(v) for k, v in m.items() if k not in ("confusion", "json_checks")}
        if isinstance(m, pd.DataFrame):
            return m.to_dict(orient="records")
        return m
    return {"engine": run.engine, "labelled_reviews": len(labels),
            "systems": {s: {"agent1": clean(E.agent1_metrics(labels, p)), "agent2": clean(E.agent2_metrics(labels, p, st.session_state.get("evd_reviews"))),
                            "agent3": clean(E.agent3_metrics(labels, p))} for s, p in run.results.items()},
            "overall": E.overall_table(labels, run.results, run.prices, st.session_state.get("evd_reviews")).to_dict(orient="records"),
            "significance": E.significance_table(labels, run.results).to_dict(orient="records"),
            "subgroups": E.subgroup_table(labels, run.results, st.session_state.get("evd_reviews") or {}).to_dict(orient="records")}
