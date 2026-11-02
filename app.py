"""BeneficiaryVoice AI — Multi-Agent Feedback Intelligence (research prototype).
صوت المستفيد الذكي — نظام متعدد الوكلاء لتحليل ملاحظات المستفيدين

Run:  streamlit run app.py

Six main pages + a compact "Reference & settings" section. Settings live in st.session_state (cfg_*),
so they apply on every page. The UI is in ui/; agents, tools, workflow and rules are in the project root.
"""
from __future__ import annotations

from datetime import date

import streamlit as st

import config
import tools
from data_loader import load_reviews
from llm import LLMClient
from ui import agent_pages as A
from ui import pages as P
from ui.eval_dashboard import page_eval_dashboard
from ui.theme import inject_css
from workflow import MultiAgentWorkflow

st.set_page_config(page_title=config.APP_NAME, page_icon=str(config.LOGO_PATH) if config.LOGO_PATH.exists() else "🏛️",
                   layout="wide", initial_sidebar_state="auto")
if config.LOGO_PATH.exists():
    st.logo(str(config.LOGO_PATH), size="large")
inject_css()


@st.cache_data(show_spinner="Loading dataset…")
def _load(data: bytes | None, path: str):
    return load_reviews(data if data is not None else path)


# --------------------------------------------------------------------------- #
# Configuration from session state (set on the Settings page), with documented defaults
# --------------------------------------------------------------------------- #
ss = st.session_state
providers = ["anthropic", "openai", "offline"]
ss.setdefault("cfg_provider", config.LLM_PROVIDER if config.LLM_PROVIDER in providers else "offline")
ss.setdefault("cfg_rev", config.MAX_REVISIONS)
ss.setdefault("cfg_date", config.SLA_REFERENCE_DATE or date.today())
ss.setdefault("cfg_presentation", False)
ss.setdefault("cfg_sla", tools.sla_targets_kb()["targets"])
ss.setdefault("cfg_matrix", tools.priority_matrix_kb()["matrix"])
try:  # hide stack traces from viewers in Management Presentation Mode
    st.set_option("client.showErrorDetails", not ss["cfg_presentation"])
except Exception:  # noqa: BLE001 - option not settable at runtime in this Streamlit version
    pass

llm = LLMClient(provider=ss["cfg_provider"])
if ss.get("llm_fatal"):  # an account/billing error seen earlier: fail fast instead of repeating it per review
    llm.fatal_error = ss["llm_fatal"]

df, info = None, None
try:
    if ss.get("dataset_bytes"):
        df, info = _load(ss["dataset_bytes"], "")
    elif config.DEFAULT_DATASET_PATH.exists():
        df, info = _load(None, str(config.DEFAULT_DATASET_PATH))
except Exception as e:  # keep the app usable with a bad upload
    st.error(f"Could not load the dataset: {e}")

ss["ctx"] = P.Ctx(llm=llm, workflow=MultiAgentWorkflow(llm, max_revisions=ss["cfg_rev"], matrix=ss["cfg_matrix"],
                                                       sla_targets=ss["cfg_sla"]),
                  df=df, info=info, ref_date=ss["cfg_date"], matrix=ss["cfg_matrix"], sla_targets=ss["cfg_sla"])

# --------------------------------------------------------------------------- #
# Navigation
# --------------------------------------------------------------------------- #
pages = {
    "data": st.Page(P.page_data_run, title="Data & Run Analysis", icon=":material/play_circle:", default=True),
    "agent1": st.Page(A.page_agent1, title="Agent 1 — Feedback Analysis", icon=":material/search:", url_path="agent1"),
    "agent2": st.Page(A.page_agent2, title="Agent 2 — Impact & Prioritization", icon=":material/balance:", url_path="agent2"),
    "agent3": st.Page(A.page_agent3, title="Agent 3 — Resolution & Escalation", icon=":material/assignment_turned_in:",
                      url_path="agent3"),
    "executive": st.Page(P.page_executive, title="Executive Dashboard", icon=":material/space_dashboard:", url_path="executive"),
    "evaluation": st.Page(page_eval_dashboard, title="Evaluation & Results", icon=":material/monitoring:", url_path="evaluation"),
    "methodology": st.Page(P.page_methodology, title="Methodology & References", icon=":material/menu_book:", url_path="methodology"),
    "settings": st.Page(P.page_settings, title="Settings", icon=":material/settings:", url_path="settings"),
}
P.PAGES.update(pages)
nav = st.navigation({"": [pages[k] for k in ("data", "agent1", "agent2", "agent3", "executive", "evaluation")],
                     "Reference & settings": [pages["methodology"], pages["settings"]]}, position="sidebar")

with st.sidebar:
    name_en, name_ar = config.APP_SHORT.split(" | ")
    st.markdown(f'<div style="font-weight:700;font-size:1.05rem">{name_en}</div>'
                f'<div dir="rtl" style="text-align:right;font-weight:700">{name_ar}</div>', unsafe_allow_html=True)
    st.caption(config.PROTOTYPE_LABEL)
    if ss.get("llm_fatal"):
        st.caption("⛔ LLM unavailable — rule-based fallback in use")
    elif llm.available:
        st.caption(f"🧠 LLM mode · {llm.model}")
    else:
        st.caption("⚙️ Rule-based mode (offline)")
    pm = st.toggle("Management Presentation Mode", value=ss["cfg_presentation"], key="w_presentation")
    if pm != ss["cfg_presentation"]:
        ss["cfg_presentation"] = pm
        st.rerun()
    st.caption("🛡️ Tickets, SLA targets and escalations are simulated.")

nav.run()
