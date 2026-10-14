"""Evaluation Dashboard: measures the evaluation plan's metrics against
human-verified reference labels (Agent 1, Agent 2, Agent 3, overall framework).

Nothing is estimated or simulated: without uploaded labels no metric is shown,
and baselines that cannot run (no LLM) are reported as not run."""
from __future__ import annotations

import hashlib
import io
import json

import pandas as pd
import streamlit as st

import baselines
import evaluation_suite as E
from ui import charts as CH
from ui import components as C

A_COLORS = {k: C.AGENTS[k]["color"] for k in ("Agent1", "Agent2", "Agent3")}
FIELD_TITLES = {"feedback_type": "Feedback type", "dimensions": "Service domain (EGSQUAL)", "sentiment": "Sentiment",
                "severity": "Severity", "priority": "Priority"}


def _f(v, fmt: str = ".3f", na: str = "n/a") -> str:
    return na if v is None or (isinstance(v, float) and pd.isna(v)) else format(v, fmt)


def _ctx():
    return st.session_state["ctx"]


# --------------------------------------------------------------------------- #
def page_eval_dashboard() -> None:
    c = _ctx()
    C.hero("Evaluation Dashboard", "Per-agent Precision / Recall / F1 against validated reference labels, ticket JSON validity, "
           "and Multi-Agent vs Single LLM vs Sequential LLM pipeline on the same test set.",
           ["Human-verified labels only", "No fabricated scores", "Cost = estimate"])
    C.mode_banner(c.llm)

    labels, ref = _step_labels(c)
    if labels is None:
        return
    run = _step_run(c, ref, labels)
    if run is None:
        return
    _results(run, labels)


# --------------------------------------------------------------------------- #
# Step 1: reference labels
# --------------------------------------------------------------------------- #
def _step_labels(c):
    C.section("1 · Reference labels (test set)", "a validated subset labelled by human annotators")
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
        st.dataframe(pd.DataFrame([{"column": k, "agent": C.agent_name(v["agent"]), "values": ", ".join(v["values"] or ["EGSQUAL keys ';'"])}
                                   for k, v in E.LABEL_SPEC.items()]), hide_index=True, width="stretch", height=320)

    up = st.file_uploader("Upload the labelled CSV (must contain `review_id`; optional `text`, `rating`, `app_id`)",
                          type=["csv"], key="evd_labels")
    if up is None:
        C.empty_state("🔒", "No reference labels → no metrics", "Precision, Recall, F1 and the system comparison are computed "
                      "only from human-verified labels. Nothing is estimated or simulated without them.")
        return None, None
    try:
        ref = pd.read_csv(io.BytesIO(up.getvalue()), encoding="utf-8-sig", dtype=str)
        labels, cov = E.normalise_reference(ref)
    except Exception as e:  # noqa: BLE001
        st.error(f"Could not read the labels: {e}")
        return None, None
    st.session_state["evd_fp"] = hashlib.sha1(up.getvalue()).hexdigest()[:12]
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
    sel = st.multiselect("Systems", options, default=options if llm_ok else ["multi_agent"], format_func=E.SYSTEMS.get,
                         key="evd_systems")
    if not llm_ok:
        st.warning("Offline mode: the Single LLM and Sequential pipeline baselines need a working LLM, so they will be "
                   "recorded as **not run** (failed), never simulated. The multi-agent framework runs with its rule-based reasoner.")
    calls = {"multi_agent": "≈3–6", "single_llm": "1", "sequential": "3"}
    st.caption(f"{len(reviews)} review(s) × {len(sel)} system(s). LLM calls per review: "
               + " · ".join(f"{E.SYSTEMS[s]} {calls[s]}" for s in sel) + (" (offline: 0)" if not llm_ok else ""))

    if st.button("▶ Run evaluation", type="primary", key="evd_run", disabled=not sel):
        run = E.EvalRun(engine=c.llm.name)
        for s in sel:
            bar = st.progress(0.0, text=f"{E.SYSTEMS[s]}: 0/{len(reviews)}")
            prog = lambda k, n, s=s, bar=bar: bar.progress(k / n, text=f"{E.SYSTEMS[s]}: {k}/{n}")  # noqa: E731
            if s == "multi_agent":
                run.results[s] = E.run_multi_agent(c.workflow, reviews, reference_date=c.ref_date, progress=prog)
            else:
                run.results[s] = baselines.run_baseline(s, c.llm, reviews, progress=prog)
            run.prices[s] = c.llm.price_per_mtok()
            bar.empty()
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
def _results(run: E.EvalRun, labels: dict) -> None:
    sys_keys = list(run.results)
    st.caption(f"Reasoning engine for this run: **{run.engine}** · predictions: "
               + ", ".join(f"{E.SYSTEMS[s]} {len(run.results[s])}" for s in sys_keys))
    show = st.segmented_control("Agent-level metrics for", sys_keys, default="multi_agent" if "multi_agent" in sys_keys else sys_keys[0],
                                format_func=E.SYSTEMS.get, key="evd_show") or sys_keys[0]
    preds = run.results[show]
    if show != "multi_agent":
        st.caption("Baselines have no separate agents; their outputs are scored on the same fields for comparison.")
    failed = sum(p.failed for p in preds)
    if failed:
        st.error(f"⛔ {failed} of {len(preds)} review(s) produced no output for {E.SYSTEMS[show]} — counted as wrong "
                 "predictions (FAILED), not removed. First error: " + next(p.error for p in preds if p.failed)[:200])
    fb = sum(p.used_fallback for p in preds)
    if fb:
        st.warning(f"{fb} review(s) used the rule-based fallback for at least one LLM step.")

    _agent1(preds, labels)
    _agent2(preds, labels)
    _agent3(preds, labels)
    _overall(run, labels)


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
        C.agent_section_header("Agent1", "categorisation & service domain vs. validated reference labels · P, R, Macro F1")
        _multiclass_block("Feedback type", m["feedback_type"], A_COLORS["Agent1"], E.FEEDBACK_TYPES, "evd_ft")
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


def _agent2(preds, labels) -> None:
    m = E.agent2_metrics(labels, preds, st.session_state.get("evd_reviews"))
    with st.container(key="frame_Agent2_evd"):
        C.agent_section_header("Agent2", "sentiment, severity and priority vs. validated reference labels · P, R, Macro F1")
        _multiclass_block("Sentiment", m["sentiment"], A_COLORS["Agent2"], E.SENTIMENTS, "evd_sent")
        with st.container(border=True):
            st.markdown("**Star rating as a supporting indicator for sentiment** (1–2★ negative · 3★ neutral · 4–5★ positive)")
            rb = m["rating_baseline"]
            C.kpi_row([("Agreement: predicted sentiment vs. rating", _f(m["rating_agreement"], ".1%"),
                        f"{m['rating_agreement_n']} review(s) with a rating", "navy"),
                       ("Rating-only Macro F1 vs. labels", _f(rb["macro_f1"]) if rb else "n/a",
                        "reference point: sentiment from stars alone", "navy"),
                       ("System Macro F1 vs. labels", _f(m["sentiment"]["macro_f1"]), "from the table above")])
            st.caption("The rating is never used as a label and never changes severity; it only helps interpret sentiment.")
        _multiclass_block("Severity", m["severity"], A_COLORS["Agent2"], E.SEVERITIES, "evd_sev")
        _multiclass_block("Priority", m["priority"], A_COLORS["Agent2"], E.PRIORITIES, "evd_pri")


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
        C.agent_section_header("Agent3", "ticket & escalation decisions vs. reference decisions · P, R, F1, JSON validity")
        a, b = st.columns(2)
        with a:
            _binary_kpis("Ticket generation decision", m["ticket"])
        with b:
            _binary_kpis("Escalation decision (simulated)", m["escalation"])
        with st.container(border=True):
            st.markdown(f"**JSON ticket validity** · {m['json_total']} generated ticket(s)")
            C.kpi_row([("JSON validity rate", _f(m["json_validity_rate"], ".1%"), f"{m['json_valid']} / {m['json_total']} valid"),
                       ("Checks", "parse · schema · fields", "allowed severity/priority · simulated + human approval", "navy")])
            bad = m["json_checks"][~m["json_checks"]["valid"]] if m["json_total"] else pd.DataFrame()
            if not bad.empty:
                st.dataframe(bad, hide_index=True, width="stretch")
        _multiclass_block("Recommended action (additional)", m["action"], A_COLORS["Agent3"], E.ACTIONS, "evd_act")


def _overall(run: E.EvalRun, labels: dict) -> None:
    tab = E.overall_table(labels, run.results, run.prices)
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
        g1, g2 = st.columns([3, 2])
        with g1, st.container(border=True):
            st.markdown("**Macro F1 per field**")
            if long.empty:
                st.caption("No labelled fields to compare.")
            else:
                st.altair_chart(CH.system_grouped_f1(long, colors), width="stretch", key="evd_ov_f1")
        with g2:
            with st.container(border=True):
                st.markdown("**Escalation F1**")
                if tab["escalation_f1"].notna().any():
                    st.altair_chart(CH.system_metric_bars(tab, "escalation_f1", "F1", colors, domain_max=1.08),
                                    width="stretch", key="evd_ov_esc")
                else:
                    st.caption("n/a — no reference escalation 'yes' cases in the labelled set.")
            with st.container(border=True):
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
            "overall": E.overall_table(labels, run.results, run.prices).to_dict(orient="records")}
