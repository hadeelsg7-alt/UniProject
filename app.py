"""Streamlit dashboard for the Multi-Agent Arabic Beneficiary Feedback Analysis prototype.

Run:  streamlit run app.py
"""
from __future__ import annotations

import html
import json
from datetime import date

import pandas as pd
import streamlit as st

import config
import evaluation
import tools
from data_loader import load_reviews, row_to_review
from llm import LLMClient
from scenarios import get_scenarios, verify_against_dataset
from schemas import ReviewInput, WorkflowState
from workflow import MultiAgentWorkflow, state_to_dict, state_to_row

st.set_page_config(page_title="Multi-Agent Arabic Feedback Analysis", page_icon="🏛️", layout="wide")

st.markdown("""
<style>
.rtl {direction: rtl; text-align: right; font-size: 1.15rem; line-height: 1.9;
      background: rgba(127,127,127,0.08); padding: 0.8rem 1rem; border-radius: 8px;}
.badge {display:inline-block; padding: 2px 10px; border-radius: 12px; font-weight: 600; margin-right: 6px;
        border: 1px solid rgba(127,127,127,0.4);}
.sev-Critical {background:#b71c1c; color:white;} .sev-High {background:#e65100; color:white;}
.sev-Medium {background:#f9a825; color:black;} .sev-Low {background:#2e7d32; color:white;}
.small {font-size: 0.85rem; opacity: 0.8;}
</style>
""", unsafe_allow_html=True)


# --------------------------------------------------------------------------- #
# Sidebar: configuration and dataset
# --------------------------------------------------------------------------- #
st.sidebar.title("⚙️ Configuration")
providers = ["anthropic", "openai", "offline"]
default_provider = config.LLM_PROVIDER if config.LLM_PROVIDER in providers else "offline"
provider = st.sidebar.selectbox("Requested reasoning engine", providers, index=providers.index(default_provider),
                                help="'offline' = transparent rule-based reasoner (no API key needed).")
llm = LLMClient(provider=provider)
if llm.available:
    st.sidebar.success(f"Using {llm.name}")
else:
    st.sidebar.warning(f"Offline rule-based reasoner. {llm.note}")
max_rev = st.sidebar.slider("Max Agent1↔Agent2 revision rounds", 0, 2, config.MAX_REVISIONS)
ref_date = st.sidebar.date_input("SLA reference date", value=config.SLA_REFERENCE_DATE or date.today(),
                                 help="Date used by the SLA checker (scenario S8 uses its own fixed date).")
st.sidebar.divider()
st.sidebar.subheader("📁 Dataset")
uploaded = st.sidebar.file_uploader("Upload reviews CSV (UTF-8)", type=["csv"])


@st.cache_data(show_spinner="Loading dataset…")
def _load(data: bytes | None, path: str):
    return load_reviews(data if data is not None else path)


df, info = None, None
try:
    if uploaded is not None:
        df, info = _load(uploaded.getvalue(), "")
    elif config.DEFAULT_DATASET_PATH.exists():
        df, info = _load(None, str(config.DEFAULT_DATASET_PATH))
except Exception as e:  # show the error but keep the app usable
    st.sidebar.error(f"Could not load dataset: {e}")
if info:
    st.sidebar.caption(f"{info['rows_available']:,} unique reviews · {info['apps']} apps · "
                       f"{info['duplicates_dropped']} duplicates removed · {info['empty_text_dropped']} empty rows removed")
    st.sidebar.caption(f"Columns used: {info['column_mapping']}")
else:
    st.sidebar.info("No dataset loaded. Upload the CSV or place it at data/reviews.csv. Scenarios still work.")

workflow = MultiAgentWorkflow(llm, max_revisions=max_rev)


# --------------------------------------------------------------------------- #
# Rendering helpers
# --------------------------------------------------------------------------- #
def rtl(text: str) -> None:
    st.markdown(f'<div class="rtl">{html.escape(text).replace(chr(10), "<br>")}</div>', unsafe_allow_html=True)


def sev_badge(sev: str | None) -> str:
    if not sev:
        return '<span class="badge">No severity</span>'
    base = sev.split(" ")[0]
    return f'<span class="badge sev-{base}">{html.escape(sev)}</span>'


def workflow_dot(state: WorkflowState) -> str:
    lines = ['digraph G {', 'rankdir=LR; node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=14, margin="0.25,0.12"];',
             'edge [fontname="Helvetica", fontsize=12]; nodesep=0.6; ranksep=1.0;',
             'R [label="Arabic review", fillcolor="#eceff1"];',
             'A1 [label="Agent 1\\nFeedback Investigation", fillcolor="#bbdefb"];',
             'A2 [label="Agent 2\\nImpact & Prioritization", fillcolor="#ffe0b2"];',
             'A3 [label="Agent 3\\nResolution & Escalation", fillcolor="#c8e6c9"];',
             'H [label="Human officer\\n(approval)", fillcolor="#f8bbd0"];']
    step = 1
    lines.append(f'R -> A1 [label="{step}. input"];')
    trace = state.node_trace
    for prev, cur in zip(trace, trace[1:]):
        step += 1
        if prev == "A1_ANALYZE" and cur == "A2_ASSESS":
            lines.append(f'A1 -> A2 [label="{step}. findings"];')
        elif prev == "A2_ASSESS" and cur == "A1_REVIEW_RESPONSE":
            lines.append(f'A2 -> A1 [label="{step}. review request", color="#d32f2f", fontcolor="#d32f2f", style=dashed];')
        elif prev == "A1_REVIEW_RESPONSE" and cur == "A2_ASSESS":
            lines.append(f'A1 -> A2 [label="{step}. revised / confirmed", color="#1565c0", fontcolor="#1565c0", style=dashed];')
        elif prev == "A2_ASSESS" and cur == "A3_RESOLVE":
            lines.append(f'A2 -> A3 [label="{step}. assessment"];')
        elif prev == "A3_RESOLVE" and cur == "END":
            lines.append(f'A3 -> H [label="{step}. simulated ticket / decision"];')
    lines.append("}")
    return "\n".join(lines)


def render_state(state: WorkflowState, key: str) -> None:
    a1, a2, a3, r = state.agent1, state.agent2, state.agent3, state.review
    if state.errors:
        with st.expander(f"⚠️ {len(state.errors)} warning(s) (e.g. LLM fallback)"):
            for e in state.errors:
                st.write(e)
    if a1 is None or a3 is None:
        st.error("The workflow did not complete. See the decision log.")
        st.dataframe(pd.DataFrame([e.model_dump() for e in state.decision_log]), width="stretch")
        return

    st.markdown("#### Original Arabic review")
    meta = f"Review `{r.review_id}` · App **{r.app_name}** (`{r.app_id}`) · Rating **{r.rating if r.rating is not None else 'n/a'}**"
    if r.is_synthetic:
        meta += " · 🧪 **SYNTHETIC**"
    st.markdown(meta)
    rtl(r.text)

    st.markdown("#### Agent interaction workflow")
    st.graphviz_chart(workflow_dot(state), width="content")
    st.caption("Path: " + " → ".join(state.node_trace) + f" · revision rounds: {state.revision_count}/{state.max_revisions}")
    with st.container():
        st.markdown("#### Final operational decision")
        st.markdown(sev_badge(a2.overall_severity.value if a2 and a2.overall_severity else None) +
                    (f'<span class="badge">{a2.overall_priority.value}</span>' if a2 and a2.overall_priority else ""),
                    unsafe_allow_html=True)
        st.info(a3.final_decision)
        st.caption(f"Reasoner: {state.reasoner}")

    # --- Agent 1 ---
    with st.container(border=True):
        st.markdown("### 🔵 Agent 1 — Feedback Investigation")
        m = st.columns(4)
        m[0].metric("Feedback type", a1.feedback_type)
        m[1].metric("Issues", len(a1.issues))
        m[2].metric("Status", a1.status.value)
        m[3].metric("Ambiguous", "Yes" if a1.is_ambiguous else "No")
        if a1.secondary_types:
            st.write("Secondary types:", ", ".join(a1.secondary_types))
        if a1.issues:
            st.dataframe(pd.DataFrame([{
                "Issue": i.issue_id, "Description": i.description, "Type": i.issue_type,
                "EGSQUAL dimension": i.dimension_name or i.dimension, "Evidence (verbatim)": i.evidence,
                "Affected service": i.affected_service, "Evidence verified": f"{i.evidence_verified} ({i.evidence_score})",
                "Missing info": "; ".join(i.missing_information), "Revision note": i.revision_note} for i in a1.issues]),
                width="stretch", hide_index=True)
        if a1.ambiguity_reasons:
            st.warning("Ambiguity: " + " | ".join(a1.ambiguity_reasons))
        if a1.missing_information:
            st.write("**Missing information:**", "; ".join(a1.missing_information))
        if a1.dialect_terms:
            st.write("**Saudi dialect detected:**", "; ".join(f"{k} → {v}" for k, v in a1.dialect_terms.items()))
        if len(state.agent1_history) > 1:
            with st.expander(f"Analysis versions ({len(state.agent1_history)})"):
                for h in state.agent1_history:
                    st.write(f"Round {h.revision_round}: type={h.feedback_type}, status={h.status.value}, issues={[i.issue_id for i in h.issues]}")

    # --- Agent 2 ---
    with st.container(border=True):
        st.markdown("### 🟠 Agent 2 — Impact Assessment & Prioritization")
        m = st.columns(4)
        m[0].metric("Overall severity", a2.overall_severity.value if a2.overall_severity else "—")
        m[1].metric("Overall priority", a2.overall_priority.value if a2.overall_priority else "—")
        m[2].metric("Verification", a2.verification_status.value)
        m[3].metric("Review requests sent", a2.review_requests_sent)
        st.write(f"**Sentiment (reported separately):** {a2.sentiment_intensity} ({a2.sentiment_score}). {a2.sentiment_vs_severity_note}")
        if a2.assessments:
            st.dataframe(pd.DataFrame([{
                "Issue": a.issue_id, "Service completion": a.service_completion, "Scope": a.scope,
                "Risk flags": ", ".join(a.risk_flags), "Impact": a.impact_summary,
                "Severity": a.severity.value + (" (provisional)" if a.provisional else ""), "Priority": a.priority.value,
                "Rules": ", ".join(a.rules_applied), "Verification": a.verification_status.value,
                "Evidence": " | ".join(a.evidence)} for a in a2.assessments]), width="stretch", hide_index=True)
            with st.expander("Reasoning per issue"):
                for a in a2.assessments:
                    st.write(f"**{a.issue_id}** — {a.reasoning}")

    # --- Agent 3 ---
    with st.container(border=True):
        st.markdown("### 🟢 Agent 3 — Resolution & Escalation")
        m = st.columns(3)
        m[0].metric("Actionable", "Yes" if a3.actionable else "No")
        m[1].metric("Escalation status", a3.escalation.status.value)
        m[2].metric("Human approval required", "Yes" if a3.human_approval_required else "No")
        st.write(f"**Action:** {a3.action.value}")
        st.write(f"**Rationale:** {a3.action_rationale}")
        if a3.secondary_actions:
            st.write("**Secondary actions:** " + "; ".join(a3.secondary_actions))
        if a3.escalation.rules:
            with st.expander("Escalation rule evaluation (PROPOSED rules - not official DGA rules)"):
                st.dataframe(pd.DataFrame([x.model_dump() for x in a3.escalation.rules]), width="stretch", hide_index=True)
                if a3.escalation.complaint_record:
                    st.json(a3.escalation.complaint_record.model_dump())
                if a3.escalation.missing_fields:
                    st.warning("Missing (not invented): " + ", ".join(a3.escalation.missing_fields))
        if a3.ticket:
            t = a3.ticket
            st.markdown(f"#### 🎫 Simulated ticket `{t.ticket_id}`")
            st.caption(t.disclaimer)
            tc1, tc2 = st.columns(2)
            with tc1:
                st.markdown(f"**Title:** {t.title}  \n**Type:** {t.ticket_type}  \n**Application:** {t.application}  \n"
                            f"**Affected service:** {t.affected_service}  \n**EGSQUAL:** {', '.join(t.egsqual_dimensions)}")
            with tc2:
                st.markdown(f"**Severity:** {t.severity}  \n**Priority:** {t.priority}  \n**Recommended action:** {t.recommended_action}  \n"
                            f"**Escalation:** {t.escalation_status}  \n**Human approval required:** {t.human_approval_required}")
            st.markdown(f"**Description:** {t.issue_description}")
            st.markdown(f"**Justification:** {t.justification}")
            with st.expander("Ticket JSON"):
                st.json(t.model_dump())

    # --- Communication, tools, log ---
    st.markdown("#### 🔁 Inter-agent communication & revision history")
    st.dataframe(pd.DataFrame([m.model_dump() for m in state.messages]), width="stretch", hide_index=True)
    t1, t2 = st.columns(2)
    with t1:
        st.markdown("#### 🛠️ Tools used")
        st.dataframe(pd.DataFrame([t.model_dump() for t in state.tool_calls]), width="stretch", hide_index=True, height=300)
    with t2:
        st.markdown("#### 📜 Decision log")
        st.dataframe(pd.DataFrame([e.model_dump() for e in state.decision_log])[["step", "agent", "action", "detail"]],
                     width="stretch", hide_index=True, height=300)

    d1, d2 = st.columns(2)
    d1.download_button("⬇️ Download full result (JSON)", json.dumps(state_to_dict(state), ensure_ascii=False, indent=2),
                       file_name=f"result_{r.review_id}.json", mime="application/json", key=f"json_{key}")
    d2.download_button("⬇️ Download summary (CSV)", pd.DataFrame([state_to_row(state)]).to_csv(index=False).encode("utf-8-sig"),
                       file_name=f"result_{r.review_id}.csv", mime="text/csv", key=f"csv_{key}")


def review_filters(prefix: str) -> pd.DataFrame | None:
    if df is None:
        return None
    f1, f2, f3 = st.columns([2, 1, 2])
    apps = f1.multiselect("Application", sorted(df["app_id"].unique()), key=f"{prefix}_apps")
    ratings = f2.multiselect("Rating", sorted(df["rating"].dropna().unique().tolist()), key=f"{prefix}_ratings")
    search = f3.text_input("Search text (Arabic)", key=f"{prefix}_search")
    sub = df
    if apps:
        sub = sub[sub["app_id"].isin(apps)]
    if ratings:
        sub = sub[sub["rating"].isin(ratings)]
    if search:
        sub = sub[sub["text"].str.contains(search, regex=False, na=False)]
    return sub


# --------------------------------------------------------------------------- #
# Layout
# --------------------------------------------------------------------------- #
st.title("🏛️ Multi-Agent AI Framework for Arabic Beneficiary Feedback Analysis")
st.caption("Research prototype · Saudi government digital services · Agents: Investigation → Impact Assessment ⇄ (revision loop) → "
           "Resolution & Escalation · All tickets and escalations are **simulated**; nothing is sent to AMER, DGA or any government system.")

if llm.available:
    st.success(f"🧠 Reasoning mode: **LLM** - `{llm.name}` (language understanding by the LLM; severity, priority and "
               "escalation decided by the documented rule engines).")
else:
    st.warning("⚙️ Reasoning mode: **OFFLINE RULE-BASED HEURISTICS** - no LLM is being called. "
               f"{llm.note} Add an API key to `.env` and restart to use the LLM.")

tab_single, tab_scen, tab_batch, tab_eval, tab_rules = st.tabs(
    ["🔎 Single review", "🧪 Test scenarios", "📦 Batch analysis", "📊 Evaluation", "📘 Rules & knowledge base"])

with tab_single:
    mode = st.radio("Input", ["Select from dataset", "Type a review"], horizontal=True)
    review = None
    if mode == "Select from dataset":
        sub = review_filters("single")
        if sub is None:
            st.info("Load a dataset from the sidebar, or use 'Type a review'.")
        elif sub.empty:
            st.warning("No reviews match the filters.")
        else:
            st.caption(f"{len(sub):,} matching reviews")
            opts = sub.head(500)
            idx = st.selectbox("Review", opts.index, format_func=lambda i: f"{opts.at[i, 'review_id']} | {opts.at[i, 'app_id']} | "
                                                                           f"★{opts.at[i, 'rating']} | {opts.at[i, 'text'][:90]}")
            review = row_to_review(opts.loc[idx])
    else:
        txt = st.text_area("Arabic review text", height=100)
        c = st.columns(3)
        app_id = c[0].text_input("App id", "CUSTOM")
        rating = c[1].selectbox("Rating", [None, 1, 2, 3, 4, 5])
        if txt.strip():
            review = ReviewInput(review_id="CUSTOM-1", text=txt.strip(), app_id=app_id, app_name=app_id, rating=rating)
    if st.button("▶️ Run Multi-Agent Analysis", type="primary", disabled=review is None):
        with st.spinner("Agents are working…"):
            st.session_state["single_state"] = workflow.run(review, reference_date=ref_date)
    if "single_state" in st.session_state:
        render_state(st.session_state["single_state"], "single")

with tab_scen:
    scen = get_scenarios()
    ver = verify_against_dataset()
    if ver["dataset_found"]:
        st.caption("Real scenario reviews verified verbatim in the dataset: " +
                   ", ".join(f"{k} {'✅' if v else '❌'}" for k, v in ver["checked"].items()))
    choice = st.selectbox("Scenario", range(len(scen)), format_func=lambda i: f"{scen[i].key} — {scen[i].title}")
    sc = scen[choice]
    st.markdown(f"**Purpose:** {sc.purpose}")
    if sc.synthetic:
        st.warning("🧪 SYNTHETIC scenario: the review text and the complaint record SYN-2026-0417 are artificial, created only to "
                   "demonstrate the escalation logic. SLA reference date fixed at " + str(sc.reference_date) + ".")
    if st.button("▶️ Run scenario", type="primary"):
        with st.spinner("Agents are working…"):
            st.session_state["scen_state"] = (sc.key, workflow.run(sc.review, reference_date=sc.reference_date or ref_date))
    if "scen_state" in st.session_state and st.session_state["scen_state"][0] == sc.key:
        render_state(st.session_state["scen_state"][1], f"scen_{sc.key}")

with tab_batch:
    sub = review_filters("batch")
    if sub is None:
        st.info("Load a dataset from the sidebar to run batch analysis.")
    else:
        c1, c2 = st.columns([1, 3])
        how = c1.radio("Selection", ["First N matching", "Random N", "Pick reviews"])
        if how == "Pick reviews":
            opts = sub.head(500)
            picked = c2.multiselect("Reviews", opts.index, format_func=lambda i: f"{opts.at[i, 'review_id']} | {opts.at[i, 'text'][:70]}")
            batch = opts.loc[picked]
        else:
            n = c2.number_input("N", 1, 2000, 20)
            batch = sub.head(int(n)) if how == "First N matching" else sub.sample(min(int(n), len(sub)), random_state=42)
        st.caption(f"{len(batch)} review(s) selected" + (" · LLM mode makes ~2-4 API calls per review" if llm.available else ""))
        if st.button("▶️ Run batch analysis", type="primary", disabled=batch.empty):
            bar = st.progress(0.0)
            states = workflow.run_batch([row_to_review(r) for _, r in batch.iterrows()], reference_date=ref_date,
                                        progress=lambda k, n: bar.progress(k / n, text=f"{k}/{n}"))
            st.session_state["batch_states"] = states
        states = st.session_state.get("batch_states")
        if states:
            rows = pd.DataFrame([state_to_row(s) for s in states])
            k = st.columns(4)
            k[0].metric("Reviews analysed", len(rows))
            k[1].metric("With revision loop", int((rows["revision_rounds"] > 0).sum()))
            k[2].metric("Need clarification", int((rows["analysis_status"] == "NEEDS_CLARIFICATION").sum()))
            k[3].metric("Escalation candidates / pending", int(rows["escalation_status"].isin(["CANDIDATE_AWAITING_HUMAN_APPROVAL", "PENDING_VERIFICATION"]).sum()))
            g = st.columns(3)
            g[0].markdown("**Severity**")
            g[0].bar_chart(rows["overall_severity"].fillna("None").value_counts())
            g[1].markdown("**Action**")
            g[1].bar_chart(rows["action"].value_counts())
            g[2].markdown("**Feedback type**")
            g[2].bar_chart(rows["feedback_type"].value_counts())
            st.markdown("**Severity vs. star rating** (shows that low ratings are not automatically severe)")
            st.dataframe(pd.crosstab(rows["rating"], rows["overall_severity"].fillna("None")), width="stretch")
            st.dataframe(rows, width="stretch", hide_index=True)
            d1, d2 = st.columns(2)
            d1.download_button("⬇️ Batch results (CSV)", rows.to_csv(index=False).encode("utf-8-sig"), "batch_results.csv", "text/csv")
            d2.download_button("⬇️ Batch results (JSON)", json.dumps([state_to_dict(s) for s in states], ensure_ascii=False, indent=1),
                               "batch_results.json", "application/json")
            pick = st.selectbox("Inspect one result", range(len(states)), format_func=lambda i: f"{states[i].review.review_id} | {states[i].review.text[:80]}")
            render_state(states[pick], f"batch_{pick}")

with tab_eval:
    st.markdown("### Scenario-based behavioural evaluation")
    st.write("Checks whether issues are supported by the review, multiple issues are found, ambiguity is flagged, severity "
             "follows the proposed rules, escalation follows the evidence, tickets contain required fields, and inter-agent "
             "review actually occurs. **These are rule-compliance checks, not accuracy scores.**")
    if st.button("▶️ Run all scenarios and evaluate", type="primary"):
        with st.spinner("Running scenarios…"):
            res = [(s.key, workflow.run(s.review, reference_date=s.reference_date or ref_date), s.expect) for s in get_scenarios()]
            st.session_state["eval_checks"] = evaluation.checks_table(res)
    if "eval_checks" in st.session_state:
        chk = st.session_state["eval_checks"]
        s = evaluation.summary(chk)
        st.success(f"Passed {s['checks_passed']} of {s['applicable']} applicable checks ({s['checks_failed']} failed, {s['not_applicable']} not applicable).")
        st.dataframe(chk, width="stretch", hide_index=True)
        st.download_button("⬇️ Checks (CSV)", chk.to_csv(index=False).encode("utf-8-sig"), "scenario_checks.csv", "text/csv")

    st.divider()
    st.markdown("### Optional: Precision / Recall / F1 against verified reference labels")
    st.write("Upload a CSV with a `review_id` column and any of: `feedback_type`, `severity`, `priority`, `action`, "
             "`escalation_status`, `dimensions` (semicolon-separated). Labels must be **verified by a human annotator**. "
             "Scores are computed against the latest **batch analysis** results only. Without labels, no scores are shown.")
    ref = st.file_uploader("Reference labels CSV", type=["csv"], key="ref_labels")
    if ref is not None:
        states = st.session_state.get("batch_states")
        if not states:
            st.warning("Run a batch analysis that includes the labelled review_ids first.")
        else:
            try:
                ref_df = pd.read_csv(ref, encoding="utf-8-sig")
                res = evaluation.prf_from_reference(ref_df, states)
                if not res:
                    st.warning("No overlapping review_ids / label columns found.")
                for field, table in res.items():
                    st.markdown(f"**{field}**")
                    st.dataframe(table, width="stretch", hide_index=True)
            except Exception as e:
                st.error(f"Could not compute metrics: {e}")
    else:
        st.info("No reference labels uploaded → Precision/Recall/F1 not computed.")

with tab_rules:
    st.warning("Severity, priority and escalation rules below are PROPOSED research rules. They are not official rules of the "
               "Saudi Digital Government Authority or of the AMER / Balaagh Digital service. Edit them in `knowledge_base/`.")
    st.markdown("### EGSQUAL service-quality dimensions (knowledge base)")
    kb = tools.egsqual_kb()
    st.dataframe(pd.DataFrame([{"key": k, "dimension": v["name_en"], "arabic": v["name_ar"], "definition": v["definition"], "source": v["source"]}
                               for k, v in kb.items()]), width="stretch", hide_index=True)
    sr = tools.severity_rules_kb()
    st.markdown("### Severity & priority rules (Agent 2)")
    st.write(sr["principle"])
    st.dataframe(pd.DataFrame(sr["severity_rules"]), width="stretch", hide_index=True)
    st.dataframe(pd.DataFrame(sr["priority_rules"]), width="stretch", hide_index=True)
    with st.expander("Impact factor definitions"):
        st.json(sr["impact_factors"])
    er = tools.escalation_rules_kb()
    st.markdown("### Escalation eligibility rules (Agent 3)")
    st.dataframe(pd.DataFrame(er["rules"]), width="stretch", hide_index=True)
    st.write(f"Minimum severity: **{er['min_severity_for_escalation']}** · default SLA: **{er['default_response_sla_days']} days** "
             f"({er['default_sla_note']}) · human approval required: **{er['require_human_approval']}**")
