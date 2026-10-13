"""Streamlit dashboard for the Multi-Agent Arabic Beneficiary Feedback Analysis prototype.

Run:  streamlit run app.py

This file only wires configuration, the sidebar and navigation. The UI lives in
ui/ (theme, components, charts, result view, pages); the research logic
(agents, tools, workflow, rules) is unchanged and lives in the project root.
"""
from __future__ import annotations

from datetime import date

import streamlit as st

import config
from data_loader import load_reviews
from llm import LLMClient
from ui import pages as P
from ui.theme import inject_css
from workflow import MultiAgentWorkflow

st.set_page_config(page_title="Multi-Agent Feedback Analysis", page_icon="🏛️", layout="wide",
                   initial_sidebar_state="auto")
inject_css()


@st.cache_data(show_spinner="Loading dataset…")
def _load(data: bytes | None, path: str):
    return load_reviews(data if data is not None else path)


# --------------------------------------------------------------------------- #
# Sidebar: reasoning mode (no secrets shown), settings, dataset
# --------------------------------------------------------------------------- #
with st.sidebar:
    st.markdown("### 🏛️ Feedback Intelligence")
    st.caption("Saudi digital government · research prototype")
    providers = ["anthropic", "openai", "offline"]
    default = config.LLM_PROVIDER if config.LLM_PROVIDER in providers else "offline"
    provider = st.selectbox("Requested reasoning engine", providers, index=providers.index(default),
                            help="The key itself is never displayed. 'offline' = transparent rule-based reasoner.")
    llm = LLMClient(provider=provider)
    if llm.available:
        st.success(f"🧠 LLM active: {llm.name}")
    else:
        st.warning("⚙️ Offline rule-based mode — no LLM call is made")
    with st.expander("Settings"):
        max_rev = st.slider("Max Agent 1 ↔ Agent 2 revision rounds", 0, 2, config.MAX_REVISIONS)
        ref_date = st.date_input("SLA reference date", value=config.SLA_REFERENCE_DATE or date.today(),
                                 help="'Today' for the SLA checker (scenario S8 uses its own fixed date).")
    st.divider()
    up = st.file_uploader("📁 Reviews dataset (CSV, UTF-8)", type=["csv"], key="sidebar_upload")
    if up is not None:
        st.session_state["dataset_bytes"] = up.getvalue()
        st.session_state["dataset_name"] = up.name

df, info = None, None
try:
    if st.session_state.get("dataset_bytes"):
        df, info = _load(st.session_state["dataset_bytes"], "")
    elif config.DEFAULT_DATASET_PATH.exists():
        df, info = _load(None, str(config.DEFAULT_DATASET_PATH))
except Exception as e:  # keep the app usable with a bad upload
    st.sidebar.error(f"Could not load dataset: {e}")
with st.sidebar:
    if info:
        st.caption(f"{st.session_state.get('dataset_name', 'data/reviews.csv')} · {info['rows_available']:,} unique reviews · "
                   f"{info['apps']} apps · {info['duplicates_dropped']} duplicates removed")
    else:
        st.caption("No dataset loaded — scenarios and typed reviews still work.")
    st.caption("🛡️ All tickets and escalations are simulated. Nothing is sent to AMER, the DGA or any government system.")

st.session_state["ctx"] = P.Ctx(llm=llm, workflow=MultiAgentWorkflow(llm, max_revisions=max_rev), df=df, info=info, ref_date=ref_date)

nav = st.navigation([
    st.Page(P.page_dashboard, title="Dashboard", icon=":material/space_dashboard:", default=True),
    st.Page(P.page_analyze, title="Analyse feedback", icon=":material/psychology:", url_path="analyse"),
    st.Page(P.page_scenarios, title="Test scenarios", icon=":material/science:", url_path="scenarios"),
    st.Page(P.page_batch, title="Batch analysis", icon=":material/stacks:", url_path="batch"),
    st.Page(P.page_evaluation, title="Evaluation", icon=":material/fact_check:", url_path="evaluation"),
    st.Page(P.page_methodology, title="Methodology & rules", icon=":material/menu_book:", url_path="methodology"),
], position="sidebar")
nav.run()
