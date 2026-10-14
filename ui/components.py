"""Reusable HTML components. Every value shown comes from real agent outputs;
all user/LLM text is HTML-escaped before rendering."""
from __future__ import annotations

import html
import re
from typing import Iterable, Optional

import streamlit as st

from ui import theme as T

esc = html.escape


def md(markup: str) -> None:
    """Render HTML through st.markdown (collapse indentation so Markdown never treats it as code)."""
    st.markdown(re.sub(r"\n\s*", "", markup), unsafe_allow_html=True)


# --------------------------------------------------------------------------- #
# Page furniture
# --------------------------------------------------------------------------- #
def hero(title: str, subtitle: str, tags: Iterable[str] = ()) -> None:
    tag_html = "".join(f'<span class="tag">{esc(t)}</span>' for t in tags)
    md(f'<div class="hero"><h1>{esc(title)}</h1><p>{esc(subtitle)}</p><div class="tagrow">{tag_html}</div></div>')


def section(title: str, hint: str = "") -> None:
    md(f'<div class="section-title"><div class="bar"></div><h3>{esc(title)}</h3>'
       f'<span class="hint">{esc(hint)}</span></div>')


def mode_banner(llm) -> None:
    if llm.available:
        md(f'<div class="mode llm"><div class="icon">🧠</div><div><b>LLM mode</b> &middot; <code>{esc(llm.name)}</code>. '
           'The LLM performs language understanding; severity, priority and escalation are decided by the documented rule '
           'engines. Any failed LLM call is flagged on the result as a <b>fallback</b>.</div></div>')
    else:
        md(f'<div class="mode offline"><div class="icon">⚙️</div><div><b>Offline rule-based mode</b> &middot; no LLM is being called. '
           f'{esc(llm.note or "")} Results come from the transparent rule-based reasoner. Add an API key to <code>.env</code> '
           'and restart to use the LLM.</div></div>')


def empty_state(icon: str, title: str, text: str) -> None:
    md(f'<div class="empty"><div class="big">{icon}</div><h4>{esc(title)}</h4><div class="small">{esc(text)}</div></div>')


def kpi_row(items: list[tuple]) -> None:
    """items: (label, value, sub, variant) with variant in '', 'navy', 'warn', 'crit'."""
    cols = st.columns(len(items))
    for col, item in zip(cols, items):
        label, value, sub, variant = (list(item) + ["", ""])[:4]
        with col:
            md(f'<div class="kpi {variant}"><div class="label">{esc(str(label))}</div>'
               f'<div class="value">{esc(str(value))}</div><div class="sub">{esc(str(sub))}</div></div>')


def pill(text: str, kind: str = "outline", title: str = "") -> str:
    t = f' title="{esc(title)}"' if title else ""
    return f'<span class="pill {kind}"{t}>{esc(text)}</span>'


# --------------------------------------------------------------------------- #
# Severity / priority indicators (colour always paired with text + icon)
# --------------------------------------------------------------------------- #
SEV_ICON = {"Low": "●", "Medium": "▲", "High": "◆", "Critical": "■"}


def severity_badge(sev: Optional[str], provisional: bool = False) -> str:
    if not sev:
        return '<span class="sev" style="background:#EDF3EF;color:#4A5E55">No severity</span>'
    return (f'<span class="sev" style="background:{T.SEVERITY_COLORS[sev]};color:{T.SEVERITY_TEXT[sev]}">'
            f'{SEV_ICON[sev]} {esc(sev)}{" (provisional)" if provisional else ""}</span>')


def priority_badge(p: Optional[str]) -> str:
    if not p:
        return '<span class="sev" style="background:#EDF3EF;color:#4A5E55">No priority</span>'
    return f'<span class="sev" style="background:{T.PRIORITY_COLORS[p]};color:{T.PRIORITY_TEXT[p]}">{esc(p)}</span>'


def severity_meter(sev: Optional[str]) -> str:
    level = T.SEVERITY_ORDER.index(sev) + 1 if sev in T.SEVERITY_ORDER else 0
    color = T.SEVERITY_COLORS.get(sev, "#E5EEE8")
    return '<div class="meter">' + "".join(
        f'<span style="background:{color if i < level else "#E5EEE8"}"></span>' for i in range(4)) + "</div>"


# --------------------------------------------------------------------------- #
# Reasoner / fallback detection (never present a fallback as an LLM result)
# --------------------------------------------------------------------------- #
def agent_reasoner(state, agent: str, llm) -> tuple[str, str]:
    """Return (label, pill-kind) describing what actually produced this agent's output."""
    fell_back = any(e.agent == agent and e.action == "LLM_FALLBACK" for e in state.decision_log)
    used_llm = any(e.agent == agent and e.action == "LLM_REASONING" for e in state.decision_log)
    if not llm.available:
        return "Rule-based (offline)", "warn"
    if fell_back and used_llm:
        return "LLM + fallback", "bad"
    if fell_back:
        return "Fallback (LLM failed)", "bad"
    if used_llm:
        return "LLM", "ok"
    return "Rules / tools only", "muted"


def fallback_banner(state, llm) -> None:
    events = [e for e in state.decision_log if e.action == "LLM_FALLBACK"]
    if llm.available and events:
        items = "".join(f"<li><b>{esc(e.agent)}</b>: {esc(e.detail[:220])}</li>" for e in events)
        md(f'<div class="mode fail"><div class="icon">⛔</div><div><b>LLM failure &middot; {len(events)} step(s) used the '
           f'rule-based fallback.</b> These parts of the result were NOT produced by the LLM:<ul style="margin:6px 0 0 0">{items}</ul></div></div>')
    elif not llm.available:
        md('<div class="mode offline"><div class="icon">⚙️</div><div>This result was produced by the <b>offline rule-based '
           'reasoner</b> (no LLM call was made).</div></div>')


# --------------------------------------------------------------------------- #
# Arabic review with highlighted evidence
# --------------------------------------------------------------------------- #
def review_with_evidence(text: str, issues: list) -> None:
    spans = []
    for k, iss in enumerate(issues):
        ev = (iss.evidence or "").strip()
        pos = text.find(ev) if ev else -1
        if pos >= 0:
            spans.append((pos, pos + len(ev), k, iss.issue_id))
    spans.sort()
    out, cur, used = [], 0, []
    for s, e, k, iid in spans:
        if s < cur:  # overlapping evidence: keep the first highlight, the card still shows the full quote
            continue
        out.append(esc(text[cur:s]))
        tint, edge = T.EVIDENCE_TINTS[k % len(T.EVIDENCE_TINTS)], T.EVIDENCE_EDGES[k % len(T.EVIDENCE_EDGES)]
        out.append(f'<mark style="background:{tint};border-color:{edge}"><sup style="color:{edge}">{esc(iid)}</sup>'
                   f'{esc(text[s:e])}</mark>')
        cur = e
        used.append(iid)
    out.append(esc(text[cur:]))
    body = "".join(out).replace("\n", "<br>")
    md(f'<div class="review-box"><div class="rtl">{body}</div></div>')
    if issues:
        st.caption("Highlighted spans are the verbatim evidence Agent 1 extracted for each issue "
                   f"({len(used)} of {len(issues)} shown inline; overlapping spans are listed on the issue cards).")


# --------------------------------------------------------------------------- #
# Workflow diagram (driven only by the real node trace)
# --------------------------------------------------------------------------- #
NODE_AGENT = {"A1_ANALYZE": "A1", "A1_REVIEW_RESPONSE": "A1", "A2_ASSESS": "A2", "A3_RESOLVE": "A3", "END": "OUT"}
NODE_LABEL = {"A1_ANALYZE": "Agent 1 analyses the review", "A2_ASSESS": "Agent 2 assesses impact",
              "A1_REVIEW_RESPONSE": "Agent 1 answers Agent 2's review request", "A3_RESOLVE": "Agent 3 decides the action",
              "END": "Decision handed to the human officer"}


def workflow_diagram(state, step: Optional[int] = None, live_node: Optional[str] = None, llm=None) -> None:
    """step: index into state.node_trace to show (None = finished run). live_node: node currently running."""
    trace = state.node_trace if state is not None else []
    if live_node is not None:
        upto = len(trace) - 1
        active_node = live_node
    elif step is not None:
        upto = step
        active_node = trace[step] if trace else None
    else:
        upto = len(trace) - 1
        active_node = None
    seen = trace[: upto + 1]
    active = NODE_AGENT.get(active_node) if active_node else None
    finished = live_node is None and step is None

    def status(agent: str) -> str:
        if agent == active and not finished:
            return "active"
        visited = any(NODE_AGENT.get(n) == agent for n in seen)
        return "done" if visited else "pending"

    a1, a2, a3 = state.agent1, state.agent2, state.agent3
    shown_a1 = state.agent1_history[min(sum(1 for n in seen if n.startswith("A1")) - 1, len(state.agent1_history) - 1)] \
        if state.agent1_history and any(n.startswith("A1") for n in seen) else None
    a1_out = (f"{esc(shown_a1.feedback_type)} &middot; {len(shown_a1.issues)} issue(s)<br>Status: {esc(shown_a1.status.value)}"
              if shown_a1 else "Waiting for input")
    a2_out = "Waiting for Agent 1"
    a2_passes = sum(1 for n in seen if n == "A2_ASSESS")
    # show Agent 2's output as it was after the pass reached at this step (never a later result)
    shown_a2 = state.agent2_history[min(a2_passes, len(state.agent2_history)) - 1] if (a2_passes and state.agent2_history) else None
    if shown_a2 is not None and status("A2") != "pending":
        tag = "<br><i>provisional – sent review request</i>" if shown_a2.reasoning_summary.startswith("[PROVISIONAL]") else ""
        if shown_a2.overall_severity:
            a2_out = (f"Severity {esc(shown_a2.overall_severity.value)} &middot; {esc(shown_a2.overall_priority.value)}"
                      f"<br>{esc(shown_a2.verification_status.value)}{tag}")
        else:
            a2_out = "No operational issue<br>No severity assigned"
    elif status("A2") == "active":
        a2_out = "Assessing impact…"
    a3_out = f"{esc(a3.action.value)}" if a3 and status("A3") == "done" else "Waiting for Agent 2"

    def badge(agent_name: str) -> str:
        if llm is None or state is None or status({"Agent1": "A1", "Agent2": "A2", "Agent3": "A3"}[agent_name]) == "pending":
            return ""
        label, kind = agent_reasoner(state, agent_name, llm)
        return f'<div style="margin-top:6px">{pill(label, kind)}</div>'

    st_lbl = {"done": "✓ Completed", "active": "● Working", "pending": "Pending"}

    def node(agent_key, title, role, out, agent_name):
        s = status(agent_key)
        if live_node is not None and s == "active":
            out = "Working on it now…"
        return (f'<div class="wf-node {s}"><div class="st">{st_lbl[s]}</div><div class="t">{title}</div>'
                f'<div class="r">{role}</div><div class="o">{out}</div>{badge(agent_name)}</div>')

    snippet = esc(state.review.text.strip()[:70]) + ("…" if len(state.review.text.strip()) > 70 else "")
    out_done = "OUT" in [NODE_AGENT.get(n) for n in seen]
    out_status = "done" if out_done else "pending"
    if a3 and out_done:
        sev_txt = (f"{a2.overall_severity.value} / {a2.overall_priority.value}" if a2 and a2.overall_severity else "no severity")
        decision = f"{esc(a3.action.value)}<br><b>{esc(sev_txt)}</b>"
    else:
        decision = "Awaiting decision"
    arrow = lambda on: f'<div class="wf-arrow{"" if on else " dim"}">➜</div>'  # noqa: E731

    rounds = sum(1 for n in seen if n == "A1_REVIEW_RESPONSE")
    requests = [m for m in state.messages if m.message_type == "REVIEW_REQUEST" and m.round <= rounds]
    if rounds:
        codes = ", ".join(sorted({m.concern_code for m in requests if m.concern_code}))
        loop = (f'<div class="lp">⟲ Agent 2 → Agent 1 revision request<span class="count">{rounds} / {state.max_revisions} round(s)</span>'
                f'<br><span style="font-weight:500">Concerns raised: {esc(codes) or "—"}</span></div>')
    else:
        loop = ('<div class="lp none">⟲ Agent 2 → Agent 1 revision loop: '
                + ("no revision was requested in this run" if finished or out_done else "not triggered so far") + "</div>")

    md('<div class="wf">'
       f'<div class="wf-node io {"done"}"><div class="st">Input</div><div class="t">Beneficiary review</div>'
       f'<div class="r">{esc(state.review.app_id)} · ★{esc(str(state.review.rating or "–"))}</div>'
       f'<div class="o" style="direction:rtl;text-align:right;font-family:IBM Plex Sans Arabic,Tahoma">{snippet}</div></div>'
       + arrow(status("A1") != "pending")
       + node("A1", "Agent 1", "Feedback Investigation", a1_out, "Agent1")
       + arrow(status("A2") != "pending")
       + node("A2", "Agent 2", "Impact & Prioritization", a2_out, "Agent2")
       + arrow(status("A3") != "pending")
       + node("A3", "Agent 3", "Resolution & Escalation", a3_out, "Agent3")
       + arrow(out_done)
       + f'<div class="wf-node io {out_status}"><div class="st">{"✓ Output" if out_done else "Output"}</div>'
         f'<div class="t">Final decision</div><div class="r">Human approval required</div><div class="o">{decision}</div></div>'
       + "</div>"
       + f'<div class="wf-loop">{loop}</div>')


# --------------------------------------------------------------------------- #
# Agent 1 issue card
# --------------------------------------------------------------------------- #
TYPE_KIND = {"Bug": "bad", "General Complaint": "warn", "Suggestion": "teal", "Inquiry": "navy", "Praise": "ok"}


def issue_card(iss, k: int) -> None:
    edge = T.EVIDENCE_EDGES[k % len(T.EVIDENCE_EDGES)]
    verified = pill("✓ Evidence verified", "ok") if iss.evidence_verified else pill("✗ Evidence not verified", "bad")
    missing = "".join(pill(m, "muted") for m in iss.missing_information) or '<span class="small">None</span>'
    note = f'<div class="small" style="margin-top:6px">✎ {esc(iss.revision_note)}</div>' if iss.revision_note else ""
    md(f'<div class="issue" style="border-left-color:{edge}"><div class="head"><div><span class="id" style="color:{edge}">{esc(iss.issue_id)}</span> '
       f'{pill(iss.issue_type, TYPE_KIND.get(iss.issue_type, "outline"))}</div><div>{verified}</div></div>'
       f'<div class="desc">{esc(iss.description)}</div>'
       f'<div class="quote">«{esc(iss.evidence)}»</div>'
       f'<div class="kv"><div class="k">EGSQUAL dimension</div><div class="v"><b>{esc(iss.dimension_name or iss.dimension)}</b>'
       f'<br><span class="small">{esc(iss.dimension_definition)}</span></div>'
       f'<div class="k">Affected service</div><div class="v">{esc(iss.affected_service)}</div>'
       f'<div class="k">Confidence</div><div class="v">{iss.confidence:.2f}</div>'
       f'<div class="k">Missing information</div><div class="v">{missing}</div></div>{note}</div>')


# --------------------------------------------------------------------------- #
# Agent 2 assessment card
# --------------------------------------------------------------------------- #
SC_KIND = {"BLOCKED": "bad", "PARTIALLY_BLOCKED": "warn", "DEGRADED": "navy", "NOT_AFFECTED": "ok"}
VS_KIND = {"VERIFIED": "ok", "VERIFIED_AFTER_REVISION": "teal", "INSUFFICIENT_EVIDENCE": "bad", "NOT_APPLICABLE": "muted"}


def assessment_card(a, issue) -> None:
    risks = "".join(pill(r, "bad") for r in a.risk_flags) or pill("No risk flag", "muted")
    rules = "".join(pill(r, "navy") for r in a.rules_applied)
    evidence = "".join(f'<div class="quote">«{esc(e)}»</div>' for e in a.evidence)
    md(f'<div class="issue" style="border-left-color:{T.SEVERITY_COLORS[a.severity.value]}"><div class="head">'
       f'<div><span class="id">{esc(a.issue_id)}</span> <span class="small">{esc(a.issue_description)}</span></div>'
       f'<div>{severity_badge(a.severity.value, a.provisional)} {priority_badge(a.priority.value)}</div></div>'
       f'{severity_meter(a.severity.value)}'
       f'<div class="kv" style="margin-top:10px"><div class="k">Service completion</div><div class="v">{pill(a.service_completion, SC_KIND[a.service_completion])}</div>'
       f'<div class="k">Scope</div><div class="v">{pill(a.scope, "outline")} {pill("workaround mentioned", "muted") if a.workaround_mentioned else ""}</div>'
       f'<div class="k">Risk flags</div><div class="v">{risks}</div>'
       f'<div class="k">Impact</div><div class="v">{esc(a.impact_summary)}</div>'
       f'<div class="k">Rules applied</div><div class="v">{rules}</div>'
       f'<div class="k">Verification</div><div class="v">{pill(a.verification_status.value, VS_KIND[a.verification_status.value])}</div>'
       f'<div class="k">Reasoning</div><div class="v small">{esc(a.reasoning)}</div></div>'
       f'<div class="small" style="margin-top:8px"><b>Evidence behind this decision</b></div>{evidence}</div>')


# --------------------------------------------------------------------------- #
# Inter-agent messages
# --------------------------------------------------------------------------- #
AGENT_COLOR = {"Agent1": "#2A78D6", "Agent2": "#C97E00", "Agent3": "#0F7A51", "Human officer": "#0E3B2C", "Orchestrator": "#4A5E55"}


def message_timeline(messages) -> None:
    if not messages:
        empty_state("💬", "No messages", "No inter-agent messages were recorded.")
        return
    rows = []
    for m in messages:
        cls = {"REVIEW_REQUEST": "req", "REVIEW_RESPONSE": "resp"}.get(m.message_type, "")
        color = AGENT_COLOR.get(m.sender, "#4A5E55")
        kind = {"REVIEW_REQUEST": "bad", "REVIEW_RESPONSE": "teal", "HANDOFF": "navy"}[m.message_type]
        extra = (pill(m.concern_code, "outline") if m.concern_code else "") + "".join(pill(i, "muted") for i in m.issue_ids)
        rnd = f" &middot; round {m.round}" if m.round else ""
        rows.append(f'<div class="msg {cls}"><div class="dot" style="background:{color}">{esc(m.sender[-1] if m.sender.startswith("Agent") else "H")}</div>'
                    f'<div class="body"><div class="meta"><b>{esc(m.sender)}</b> → <b>{esc(m.receiver)}</b>{rnd} {pill(m.message_type, kind)} {extra}</div>'
                    f'<div>{esc(m.content)}</div></div></div>')
    md("".join(rows))


# --------------------------------------------------------------------------- #
# Agent 3: escalation rules + ticket
# --------------------------------------------------------------------------- #
RULE_ICON = {"MET": ("✓", "#0CA30C"), "NOT_MET": ("✗", "#D03B3B"), "UNKNOWN": ("?", "#C97E00")}


def rules_checklist(esc_assessment) -> None:
    rows = []
    for r in esc_assessment.rules:
        icon, color = RULE_ICON[r.result]
        rows.append(f'<div class="rule"><div class="ic" style="color:{color}">{icon}</div><div><b>{esc(r.rule_id)}</b> '
                    f'{pill(r.result, {"MET": "ok", "NOT_MET": "bad", "UNKNOWN": "warn"}[r.result])} {esc(r.description)}'
                    f'<div class="small">{esc(r.evidence)}</div></div></div>')
    md('<div class="card">' + "".join(rows) + "</div>")


def ticket_card(t) -> None:
    fields = [("Ticket type", t.ticket_type), ("Application", t.application), ("Affected service", t.affected_service),
              ("Severity", t.severity), ("Priority", t.priority), ("Escalation status", t.escalation_status),
              ("EGSQUAL dimensions", ", ".join(t.egsqual_dimensions)), ("Source review", t.original_review_id),
              ("Human approval", "Required" if t.human_approval_required else "Not required"), ("Created", t.created_at)]
    grid = "".join(f'<div class="f"><div class="k">{esc(k)}</div><div class="v">{esc(str(v))}</div></div>' for k, v in fields)
    ev = "".join(f'<div class="quote">«{esc(e)}»</div>' for e in t.evidence)
    md(f'<div class="ticket"><div class="th"><div><span class="id">{esc(t.ticket_id)}</span><div style="font-weight:700;font-size:1.05rem;margin-top:2px">'
       f'{esc(t.title)}</div></div><span class="sim">SIMULATED · NOT SUBMITTED</span></div><div class="tb">'
       f'<div class="grid">{grid}</div>'
       f'<div class="small"><b>Description</b></div><div style="margin-bottom:8px">{esc(t.issue_description)}</div>'
       f'<div class="small"><b>Recommended action</b></div><div style="margin-bottom:8px">{esc(t.recommended_action)}</div>'
       f'<div class="small"><b>Justification</b></div><div style="margin-bottom:8px">{esc(t.justification)}</div>'
       f'<div class="small"><b>Beneficiary evidence</b></div>{ev}</div>'
       f'<div class="foot">⚠️ {esc(t.disclaimer)}</div></div>')
