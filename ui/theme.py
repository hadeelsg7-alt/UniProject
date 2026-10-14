"""Design tokens and global CSS for the dashboard (forest green / white / emerald).
Constant names NAVY/TEAL are kept for compatibility; they now hold the green brand colours."""
from __future__ import annotations

import streamlit as st

NAVY = "#0E3B2C"
NAVY_2 = "#155A40"
TEAL = "#0F7A51"
TEAL_LIGHT = "#E2F4EA"
INK = "#10261D"
INK_2 = "#4A5E55"
MUTED = "#75887F"
SURFACE = "#FFFFFF"
BG = "#F3F8F5"
BORDER = "#D6E5DC"

# Status palette (validated; always paired with a text label / icon)
SEVERITY_COLORS = {"Low": "#0CA30C", "Medium": "#FAB219", "High": "#EC835A", "Critical": "#D03B3B"}
SEVERITY_TEXT = {"Low": "#FFFFFF", "Medium": "#1B1B1B", "High": "#1B1B1B", "Critical": "#FFFFFF"}
SEVERITY_ORDER = ["Low", "Medium", "High", "Critical"]
# Ordinal single-hue ramp for priority (P4 light -> P1 dark); validated --ordinal
PRIORITY_COLORS = {"P4": "#86B6EF", "P3": "#3987E5", "P2": "#1C5CAB", "P1": "#0D366B"}
PRIORITY_TEXT = {"P4": "#10261D", "P3": "#FFFFFF", "P2": "#FFFFFF", "P1": "#FFFFFF"}
PRIORITY_ORDER = ["P1", "P2", "P3", "P4"]
# Three-series categorical (validated all-pairs) for Precision / Recall / F1
METRIC_COLORS = {"precision": "#2A78D6", "recall": "#EB6834", "f1": "#1BAF7A"}
# Evidence highlight colours per issue (background tints, text stays ink)
EVIDENCE_TINTS = ["#D3F0DF", "#FFE7C2", "#DCE8FB", "#F6D9E6", "#CDEEF0", "#EDE3FA"]
EVIDENCE_EDGES = ["#0F7A51", "#C97E00", "#2A78D6", "#C2477A", "#0E7C86", "#6250D6"]

CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans+Arabic:wght@400;500;600&family=Inter:wght@400;500;600;700&display=swap');
:root {{ --navy:{NAVY}; --navy2:{NAVY_2}; --teal:{TEAL}; --teal-l:{TEAL_LIGHT}; --ink:{INK}; --ink2:{INK_2};
         --muted:{MUTED}; --surface:{SURFACE}; --bg:{BG}; --border:{BORDER}; }}
html, body, [class*="css"], .stMarkdown, .stText {{ font-family: 'Inter', 'IBM Plex Sans Arabic', system-ui, sans-serif; }}
.block-container {{ padding-top: 1.6rem; padding-bottom: 3rem; max-width: 1400px; }}
h1, h2, h3, h4 {{ color: var(--navy); letter-spacing: -0.01em; }}

/* Sidebar */
section[data-testid="stSidebar"] {{ background: linear-gradient(180deg, var(--navy) 0%, var(--navy2) 100%); }}
section[data-testid="stSidebar"] * {{ color: #E6F1EB !important; }}
section[data-testid="stSidebar"] [data-baseweb="select"] * , section[data-testid="stSidebar"] input {{ color: var(--ink) !important; }}
section[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] {{ background: rgba(255,255,255,0.06); border-color: rgba(255,255,255,0.25); }}
section[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] button * {{ color: var(--ink) !important; }}
section[data-testid="stSidebar"] a[data-testid="stSidebarNavLink"] {{ border-radius: 8px; }}
section[data-testid="stSidebar"] a[data-testid="stSidebarNavLink"][aria-current="page"] {{ background: rgba(15,122,81,0.35); }}

/* Hero */
.hero {{ background: linear-gradient(120deg, var(--navy) 0%, var(--navy2) 55%, var(--teal) 140%); color: #fff;
        border-radius: 18px; padding: 26px 30px; margin-bottom: 18px; box-shadow: 0 10px 30px rgba(14,59,44,0.18); }}
.hero h1 {{ color: #fff; font-size: 1.75rem; margin: 0 0 6px 0; line-height: 1.25; }}
.hero p {{ color: #CDE6D8; margin: 0; font-size: 0.98rem; }}
.hero .tagrow {{ margin-top: 12px; display:flex; gap:8px; flex-wrap:wrap; }}
.hero .tag {{ background: rgba(255,255,255,0.12); border:1px solid rgba(255,255,255,0.25); color:#fff; padding:3px 10px; border-radius:999px; font-size:0.78rem; }}

/* Mode banner */
.mode {{ border-radius: 12px; padding: 12px 16px; margin: 4px 0 16px 0; display:flex; gap:12px; align-items:flex-start; font-size:0.93rem; }}
.mode .icon {{ font-size: 1.3rem; line-height: 1.2; }}
.mode.llm {{ background:#E7F6EC; border:1px solid #9AD3AE; color:#0E4D24; }}
.mode.offline {{ background:#FFF6E0; border:1px solid #F2CF73; color:#5C4300; }}
.mode.fail {{ background:#FDECEC; border:1px solid #E9A3A3; color:#7A1414; }}
.mode b {{ font-weight: 700; }}

/* Cards */
.card {{ background: var(--surface); border: 1px solid var(--border); border-radius: 14px; padding: 16px 18px;
        box-shadow: 0 1px 2px rgba(14,59,44,0.04); margin-bottom: 12px; }}
.card h4 {{ margin: 0 0 8px 0; font-size: 1rem; }}
.kpi {{ background: var(--surface); border:1px solid var(--border); border-radius: 14px; padding: 14px 16px; height: 100%;
       border-top: 4px solid var(--teal); box-shadow: 0 1px 2px rgba(14,59,44,0.05); }}
.kpi .label {{ color: var(--ink2); font-size: 0.78rem; text-transform: uppercase; letter-spacing: .05em; font-weight:600; }}
.kpi .value {{ color: var(--navy); font-size: 1.85rem; font-weight: 700; line-height: 1.15; margin-top: 4px; }}
.kpi .sub {{ color: var(--muted); font-size: 0.8rem; margin-top: 2px; }}
.kpi.navy {{ border-top-color: var(--navy); }} .kpi.warn {{ border-top-color: #EC835A; }} .kpi.crit {{ border-top-color: #D03B3B; }}
.section-title {{ display:flex; align-items:center; gap:10px; margin: 18px 0 10px 0; }}
.section-title .bar {{ width: 5px; height: 22px; background: var(--teal); border-radius: 3px; }}
.section-title h3 {{ margin: 0; font-size: 1.15rem; }}
.section-title .hint {{ color: var(--muted); font-size: 0.85rem; }}

/* Pills / badges */
.pill {{ display:inline-flex; align-items:center; gap:5px; padding: 3px 10px; border-radius: 999px; font-size: 0.78rem;
        font-weight: 600; border: 1px solid transparent; white-space: nowrap; margin: 2px 4px 2px 0; }}
.pill.outline {{ background: #fff; border-color: var(--border); color: var(--ink2); }}
.pill.teal {{ background: var(--teal-l); color: #0B5E3D; border-color:#B5DFC7; }}
.pill.navy {{ background: #E3EFE8; color: var(--navy); border-color:#C5DACE; }}
.pill.ok {{ background:#E7F6EC; color:#0E5A26; border-color:#A8DDB8; }}
.pill.warn {{ background:#FFF3D6; color:#6B4D00; border-color:#F2CF73; }}
.pill.bad {{ background:#FDECEC; color:#8A1C1C; border-color:#E9A3A3; }}
.pill.muted {{ background:#F0F5F2; color:var(--ink2); border-color:var(--border); }}

/* Arabic RTL */
.rtl {{ direction: rtl; text-align: right; font-family: 'IBM Plex Sans Arabic', 'Geeza Pro', 'Segoe UI', Tahoma, sans-serif;
       font-size: 1.22rem; line-height: 2.1; color: var(--ink); }}
.review-box {{ background: #FBFDFC; border: 1px solid var(--border); border-right: 5px solid var(--teal); border-radius: 12px; padding: 14px 18px; }}
.rtl mark {{ padding: 2px 4px; border-radius: 5px; border-bottom: 2px solid; color: var(--ink); }}
.rtl mark sup {{ font-family: Inter, sans-serif; font-size: 0.65rem; font-weight: 700; margin-right: 3px; }}
.quote {{ direction: rtl; text-align:right; font-family:'IBM Plex Sans Arabic', Tahoma, sans-serif; font-size: 1.05rem; line-height: 1.9;
         background: #F3FAF6; border-right: 4px solid var(--teal); border-radius: 8px; padding: 8px 12px; margin: 6px 0; }}

/* Issue / assessment cards */
.issue {{ background:#fff; border:1px solid var(--border); border-left: 5px solid var(--teal); border-radius: 12px; padding: 14px 16px; margin-bottom: 12px; }}
.issue .head {{ display:flex; justify-content:space-between; align-items:center; gap: 10px; flex-wrap: wrap; }}
.issue .id {{ font-weight: 700; color: var(--navy); }}
.issue .desc {{ font-weight: 600; color: var(--ink); margin: 6px 0; font-size: 1rem; }}
.kv {{ display:grid; grid-template-columns: 150px 1fr; gap: 4px 12px; font-size: 0.88rem; margin-top: 6px; }}
.kv .k {{ color: var(--muted); }} .kv .v {{ color: var(--ink); }}
.small {{ font-size: 0.82rem; color: var(--ink2); }}

/* Severity / priority indicators */
.sev {{ display:inline-flex; align-items:center; gap:6px; padding: 4px 12px; border-radius: 8px; font-weight: 700; font-size: 0.85rem; }}
.meter {{ display:flex; gap:4px; margin-top: 6px; }}
.meter span {{ flex:1; height: 8px; border-radius: 4px; background:#E5EEE8; }}

/* Workflow diagram */
.wf {{ display:flex; align-items:stretch; gap: 0; flex-wrap: nowrap; justify-content:center; margin: 14px 0 4px 0; }}
.wf-node {{ background:#fff; border:2px solid var(--border); border-radius: 14px; padding: 12px 12px; flex: 1 1 190px; max-width: 215px; min-width: 0; min-height: 128px;
           position: relative; transition: all .2s; }}
.wf-node.io {{ flex: 0.8 1 140px; max-width: 165px; background: #F5FAF7; }}
.wf-node .t {{ font-weight: 700; color: var(--navy); font-size: 0.95rem; }}
.wf-node .r {{ color: var(--muted); font-size: 0.75rem; margin-bottom: 6px; }}
.wf-node .o {{ font-size: 0.8rem; color: var(--ink); line-height: 1.45; }}
.wf-node .st {{ position:absolute; top: -11px; right: 10px; font-size: 0.7rem; font-weight: 700; padding: 2px 8px; border-radius: 999px; }}
.wf-node.done {{ }}
.wf-node.done .st {{ background: var(--teal); color:#fff; }}
.wf-node.active {{ box-shadow: 0 0 0 4px rgba(15,122,81,0.18), 0 8px 22px rgba(14,59,44,0.12); transform: translateY(-2px); }}
.wf-node.active .st {{ background: #FAB219; color:#1B1B1B; animation: pulse 1.2s infinite; }}
.wf-node.pending {{ opacity: 0.55; }}
.wf-node.pending .st {{ background: #E5EEE8; color: var(--ink2); }}
@keyframes pulse {{ 0% {{ opacity: 1; }} 50% {{ opacity: .55; }} 100% {{ opacity: 1; }} }}
.wf-arrow {{ display:flex; align-items:center; justify-content:center; flex: 0 0 30px; color: var(--teal); font-size: 1.6rem; font-weight: 700; }}
.wf-arrow.dim {{ color: #C2D1C8; }}
.wf-loop {{ display:flex; justify-content:center; margin-top: 6px; }}
.wf-loop .lp {{ border: 2px dashed #D03B3B; border-top: none; border-radius: 0 0 16px 16px; padding: 6px 18px 8px 18px; color:#8A1C1C;
               font-size: 0.83rem; font-weight: 600; background: #FFF7F7; text-align:center; min-width: 380px; }}
.wf-loop .lp.none {{ border-color: #C2D1C8; color: var(--muted); background: #F9FCFA; font-weight: 500; }}
.wf-loop .count {{ display:inline-block; background:#D03B3B; color:#fff; border-radius: 999px; padding: 0 9px; margin-left: 6px; }}

/* Agent identity */
.agent-row {{ display:flex; align-items:stretch; gap:0; margin: 4px 0 14px 0; }}
.agent-card {{ flex:1; background:#fff; border:1px solid var(--border); border-top:5px solid var(--teal); border-radius:14px; padding:12px 14px; }}
.agent-arrow {{ display:flex; align-items:center; padding: 0 8px; color: var(--teal); font-size:1.4rem; font-weight:700; }}
.agent-tag {{ display:inline-block; color:#fff; font-size:0.68rem; font-weight:800; letter-spacing:.08em; padding:2px 8px; border-radius:6px; margin-bottom:4px; }}
.agent-title {{ font-weight:700; color: var(--navy); font-size:1.02rem; }}
.agent-ar {{ direction: rtl; text-align: right; font-family:'IBM Plex Sans Arabic', Tahoma, sans-serif; color: var(--ink2); font-size: 0.95rem; }}
.agent-ar.small-ar {{ font-size: 0.85rem; text-align: left; direction: rtl; unicode-bidi: plaintext; }}
/* Per-agent frames: everything an agent produced sits inside its own coloured frame */
[class*="st-key-frame_Agent1"] {{ --agent: #2A78D6; --agent-bg: #F3F8FE; }}
[class*="st-key-frame_Agent2"] {{ --agent: #C97E00; --agent-bg: #FFF9EE; }}
[class*="st-key-frame_Agent3"] {{ --agent: #0F7A51; --agent-bg: #F1F9F4; }}
[class*="st-key-frame_Agent"] {{ border: 3px solid var(--agent) !important; border-radius: 18px !important;
                                background: var(--agent-bg) !important; padding: 14px 16px 6px 16px !important; }}
[class*="st-key-frame_Agent"] .issue,
[class*="st-key-frame_Agent"] .card,
[class*="st-key-frame_Agent"] .decision,
[class*="st-key-frame_Agent"] .ticket {{ border-top: 2px solid var(--agent) !important; border-right: 2px solid var(--agent) !important;
                                        border-bottom: 2px solid var(--agent) !important; }}
[class*="st-key-frame_Agent"] .card:not([style*="border-left"]),
[class*="st-key-frame_Agent"] .decision,
[class*="st-key-frame_Agent"] .ticket {{ border-left: 2px solid var(--agent) !important; }}
[class*="st-key-frame_Agent"] .ticket .th {{ background: var(--agent) !important; }}
[class*="st-key-frame_Agent"] .kpi {{ border-top-color: var(--agent) !important; border-color: var(--agent); }}
/* Messages timeline */
.msg {{ display:flex; gap: 12px; margin-bottom: 10px; }}
.msg .dot {{ width: 36px; height: 36px; font-size: 1.05rem; border-radius: 50%; display:flex; align-items:center; justify-content:center; color:#fff;
            font-weight: 700; font-size: 0.8rem; flex-shrink: 0; }}
.msg .body {{ background:#fff; border:1px solid var(--border); border-radius: 12px; padding: 8px 12px; flex: 1; }}
.msg .meta {{ font-size: 0.76rem; color: var(--muted); margin-bottom: 2px; }}
.msg.req .body {{ border-color: #E9A3A3; background: #FFF8F8; }}
.msg.resp .body {{ border-color: #A3D6B8; background: #F2FAF5; }}

/* Ticket */
.ticket {{ background:#fff; border:1px solid var(--border); border-radius: 14px; overflow:hidden; position: relative; }}
.ticket .th {{ background: var(--navy); color:#fff; padding: 12px 18px; display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:8px; }}
.ticket .th .id {{ font-family: ui-monospace, Menlo, monospace; font-size: 0.9rem; }}
.ticket .sim {{ background:#FAB219; color:#1B1B1B; font-weight:800; font-size:0.72rem; padding: 3px 10px; border-radius: 6px; letter-spacing:.06em; }}
.ticket .tb {{ padding: 14px 18px; }}
.ticket .grid {{ display:grid; grid-template-columns: repeat(auto-fit, minmax(170px, 1fr)); gap: 10px 18px; margin: 8px 0 12px 0; }}
.ticket .f .k {{ font-size: 0.72rem; color: var(--muted); text-transform: uppercase; letter-spacing: .05em; }}
.ticket .f .v {{ font-size: 0.92rem; color: var(--ink); font-weight: 600; }}
.ticket .foot {{ background:#FFF8E6; border-top:1px dashed #F2CF73; color:#5C4300; font-size: 0.8rem; padding: 8px 18px; }}

/* Rule checklist */
.rule {{ display:flex; gap:10px; align-items:flex-start; padding: 7px 0; border-bottom: 1px solid #EDF3EF; font-size: 0.88rem; }}
.rule:last-child {{ border-bottom: none; }}
.rule .ic {{ width: 22px; text-align:center; font-weight: 700; }}
.decision {{ border-radius: 14px; padding: 16px 18px; background: linear-gradient(120deg, #EEF8F2, #FFFFFF); border:1px solid #B5DFC7; }}
.decision .act {{ font-size: 1.25rem; font-weight: 700; color: var(--navy); }}
.empty {{ text-align:center; color: var(--ink2); background:#fff; border:1px dashed #C2D1C8; border-radius: 14px; padding: 28px 18px; }}
.empty .big {{ font-size: 1.8rem; }}
textarea {{ unicode-bidi: plaintext; text-align: start; font-size: 1.05rem !important; }}
@media (max-width: 900px) {{ .agent-row {{ flex-direction: column; }} .agent-arrow {{ transform: rotate(90deg); justify-content:center; }} .wf {{ flex-wrap: wrap; }} .wf-node, .wf-node.io {{ max-width: 100%; flex-basis: 100%; }} .wf-arrow {{ width:100%; transform: rotate(90deg); height: 30px; }}
                             .wf-loop .lp {{ min-width: 0; width: 100%; }} .kv {{ grid-template-columns: 1fr; }} }}
</style>
"""


def inject_css() -> None:
    st.markdown(CSS, unsafe_allow_html=True)
