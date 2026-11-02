"""GovFeedback AI — Multi-Agent Intelligence for Digital Service Quality (research prototype).

Run:  streamlit run app.py

This file wires the sidebar (seven pages + a compact Settings section) and the shared context.
The UI lives in ui/; the agents, tools, workflow and research rules live in the project root and
knowledge_base/.
"""
from __future__ import annotations

from datetime import date

import pandas as pd
import streamlit as st

import config
import tools
from data_loader import load_reviews
from llm import LLMClient
from ui import agent_pages as A
from ui import pages as P
from ui.theme import inject_css
from workflow import MultiAgentWorkflow

st.set_page_config(page_title="GovFeedback AI", page_icon="🏛️", layout="wide", initial_sidebar_state="auto")
inject_css()
LEVELS = ["High", "Medium", "Low"]
PRIOS = ["P1", "P2", "P3", "P4"]


@st.cache_data(show_spinner="Loading dataset…")
def _load(data: bytes | None, path: str):
    return load_reviews(data if data is not None else path)


# --------------------------------------------------------------------------- #
# Navigation: seven pages
# --------------------------------------------------------------------------- #
pages = {
    "executive": st.Page(P.page_executive, title="Executive Dashboard", icon=":material/space_dashboard:", default=True),
    "agent1": st.Page(A.page_agent1, title="Agent 1 — Feedback Analysis", icon=":material/search:", url_path="agent1"),
    "agent2": st.Page(A.page_agent2, title="Agent 2 — Impact & Prioritization", icon=":material/balance:", url_path="agent2"),
    "agent3": st.Page(A.page_agent3, title="Agent 3 — Resolution & Escalation", icon=":material/assignment_turned_in:",
                      url_path="agent3"),
    "results": st.Page(P.page_results, title="Results & Evaluation", icon=":material/monitoring:", url_path="results"),
    "insights": st.Page(P.page_insights, title="Service Quality Insights", icon=":material/insights:", url_path="insights"),
    "methodology": st.Page(P.page_methodology, title="Methodology & Rules", icon=":material/menu_book:", url_path="methodology"),
}
P.PAGES.update(pages)
nav = st.navigation(list(pages.values()), position="sidebar")

# --------------------------------------------------------------------------- #
# Sidebar: brand, then a compact Settings section (no secret is ever displayed)
# --------------------------------------------------------------------------- #
with st.sidebar:
    st.markdown("**🏛️ GovFeedback AI**")
    st.caption("Multi-Agent Intelligence for Digital Service Quality · Research Prototype — Proposed Decision-Support Solution")
    providers = ["anthropic", "openai", "offline"]
    default = config.LLM_PROVIDER if config.LLM_PROVIDER in providers else "offline"
    provider = st.session_state.get("set_provider", default)
    llm = LLMClient(provider=provider)
    st.caption(("🧠 LLM active: " + llm.name) if llm.available else "⚙️ Offline rule-based mode")

    with st.expander("⚙️ Settings", expanded=False):
        st.selectbox("Reasoning engine (AI model)", providers, index=providers.index(provider), key="set_provider",
                     help="The API key is read from .env and never displayed. 'offline' = transparent rule-based reasoner.")
        s = config.llm_settings_summary()
        st.caption(f"Anthropic model: {s['anthropic_model']} · key {'set' if s['anthropic_key_set'] else 'not set'}  \n"
                   f"OpenAI model: {s['openai_model']} · key {'set' if s['openai_key_set'] else 'not set'}")
        if llm.note:
            st.caption("⚠️ " + llm.note)
        max_rev = st.slider("Max Agent 1 ↔ Agent 2 review rounds", 0, 2, config.MAX_REVISIONS, key="set_rev")
        ref_date = st.date_input("SLA reference date", value=config.SLA_REFERENCE_DATE or date.today(), key="set_date",
                                 help="'Today' for the complaint-SLA check (scenario S8 uses its own fixed date).")
        up = st.file_uploader("📁 Reviews dataset (CSV, UTF-8)", type=["csv"], key="sidebar_upload")
        if up is not None:
            st.session_state["dataset_bytes"] = up.getvalue()
            st.session_state["dataset_name"] = up.name

        st.markdown("**Simulated SLA targets** (research-defined, hours)")
        sla_default = pd.DataFrame([{"priority": p, "response_h": v["response_hours"], "resolution_h": v["resolution_hours"]}
                                    for p, v in tools.sla_targets_kb()["targets"].items()])
        sla_df = st.data_editor(sla_default, key="set_sla", hide_index=True, disabled=["priority"], width="stretch",
                                column_config={"response_h": st.column_config.NumberColumn(min_value=1, step=1),
                                               "resolution_h": st.column_config.NumberColumn(min_value=1, step=1)})
        st.markdown("**Impact × Urgency matrix** (rows = impact, columns = urgency)")
        m_default = pd.DataFrame([{"impact": i, **tools.priority_matrix_kb()["matrix"][i]} for i in LEVELS])
        m_df = st.data_editor(m_default, key="set_matrix", hide_index=True, disabled=["impact"], width="stretch",
                              column_config={u: st.column_config.SelectboxColumn(u, options=PRIOS, required=True) for u in LEVELS})
        st.caption("Changes apply to reviews analysed from now on; earlier results keep the settings they were analysed with.")
    st.caption("🛡️ Tickets, SLA targets and escalations are simulated. Nothing is sent to AMER, the DGA or any government system.")

sla_targets = {r.priority: {"response_hours": int(r.response_h), "resolution_hours": int(max(r.resolution_h, r.response_h))}
               for r in sla_df.itertuples()}
matrix = {r.impact: {u: getattr(r, u) for u in LEVELS} for r in m_df.itertuples()}

df, info = None, None
try:
    if st.session_state.get("dataset_bytes"):
        df, info = _load(st.session_state["dataset_bytes"], "")
    elif config.DEFAULT_DATASET_PATH.exists():
        df, info = _load(None, str(config.DEFAULT_DATASET_PATH))
except Exception as e:  # keep the app usable with a bad upload
    st.sidebar.error(f"Could not load dataset: {e}")
if info:
    st.sidebar.caption(f"Dataset: {st.session_state.get('dataset_name', 'data/reviews.csv')} · {info['rows_available']:,} unique reviews · "
                       f"{info['apps']} apps")

st.session_state["ctx"] = P.Ctx(llm=llm, workflow=MultiAgentWorkflow(llm, max_revisions=max_rev, matrix=matrix, sla_targets=sla_targets),
                                df=df, info=info, ref_date=ref_date, matrix=matrix, sla_targets=sla_targets)
nav.run()
