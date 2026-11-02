"""Service-quality insights computed from real workflow results.

One function, `compute_insights`, turns the analysed WorkflowStates into every
number, table and recommendation shown on the Service Quality Insights page and
in the executive PDF. Both read the same dict, so the page and the report cannot
disagree. Nothing is estimated: every value is a count over the analysed results,
and every recommendation states the counts it is based on.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from datetime import date
from typing import Optional

import lexicon as lx
import tools
from evaluation_suite import sentiment_class
from schemas import SEVERITY_ORDER, ActionType, EscalationStatus, WorkflowState

PRIORITIES = ["P1", "P2", "P3", "P4"]
SEVERITIES = ["Critical", "High", "Medium", "Low"]
SENTIMENTS = ["Negative", "Neutral", "Positive"]
FEEDBACK_TYPES = ["Bug", "General Complaint", "Suggestion", "Inquiry", "Praise"]
URGENT = {"P1", "P2"}
PROTOTYPE_LABEL = "Research Prototype — Proposed Decision-Support Solution"


def _pct(n: int, d: int) -> float:
    return round(100 * n / d, 1) if d else 0.0


def _topic_label(topic: str) -> str:
    return lx.TOPICS.get(topic, lx.GENERIC_TOPIC).label_en


def issue_rows(states: list[WorkflowState]) -> list[dict]:
    """One row per assessed issue, joining Agent 1's issue, Agent 2's assessment and Agent 3's action."""
    rows = []
    for s in states:
        if not (s.agent1 and s.agent2 and s.agent3):
            continue
        issues = {i.issue_id: i for i in s.agent1.issues}
        for a in s.agent2.assessments:
            i = issues.get(a.issue_id)
            if i is None:
                continue
            rows.append({
                "review_id": s.review.review_id, "app": s.review.app_name or s.review.app_id, "issue_id": i.issue_id,
                "description": i.description, "issue_type": i.issue_type, "topic": i.topic, "topic_label": _topic_label(i.topic),
                "dimension": i.dimension, "dimension_name": i.dimension_name or i.dimension, "egsqual": i.egsqual_candidate,
                "dimension_reason": i.dimension_reason, "evidence": i.evidence, "severity": a.severity.value,
                "priority": a.priority.value, "provisional": a.provisional, "action": s.agent3.action.value,
            })
    return rows


def compute_insights(states: list[WorkflowState], analysis_date: Optional[date] = None) -> dict:
    done = [s for s in states if s.agent1 and s.agent2 and s.agent3]
    rows = issue_rows(done)
    n_rev, n_iss = len(done), len(rows)
    kb = tools.egsqual_kb()

    apps = Counter(s.review.app_name or s.review.app_id for s in done)
    types = Counter(s.agent1.feedback_type for s in done)
    senti = Counter(sentiment_class(s.agent2.sentiment_intensity) for s in done)
    sev = Counter(r["severity"] for r in rows)
    prio = Counter(r["priority"] for r in rows)
    actions = Counter(s.agent3.action.value for s in done)
    esc = Counter(s.agent3.escalation.status.value for s in done)

    # ---- EGSQUAL / service-quality dimensions --------------------------------
    by_dim: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_dim[r["dimension"]].append(r)
    dimensions = []
    for key, rs in sorted(by_dim.items(), key=lambda kv: -len(kv[1])):
        app_counts = Counter(r["app"] for r in rs)
        dimensions.append({
            "key": key, "name": kb.get(key, {}).get("name_en", key), "name_ar": kb.get(key, {}).get("name_ar", ""),
            "egsqual": kb.get(key, {}).get("egsqual_candidate"), "issues": len(rs), "share": _pct(len(rs), n_iss),
            "high_critical": sum(r["severity"] in ("High", "Critical") for r in rs),
            "urgent": sum(r["priority"] in URGENT for r in rs),
            "top_apps": app_counts.most_common(3),
            "examples": [{"review_id": r["review_id"], "app": r["app"], "evidence": r["evidence"]}
                         for r in sorted(rs, key=lambda r: (r["priority"], -SEVERITY_ORDER[r["severity"]]))[:2]],
        })

    # ---- Most frequent issues: grouped by dimension + topic -------------------
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for r in rows:
        groups[(r["dimension"], r["topic"])].append(r)
    top_issues = []
    for (dim, topic), rs in sorted(groups.items(), key=lambda kv: (-len(kv[1]), kv[0])):
        worst = min(rs, key=lambda r: (r["priority"], -SEVERITY_ORDER[r["severity"]]))
        top_issues.append({
            "issue": _topic_label(topic), "dimension": kb.get(dim, {}).get("name_en", dim), "count": len(rs),
            "reviews": len({r["review_id"] for r in rs}), "apps": [a for a, _ in Counter(r["app"] for r in rs).most_common(3)],
            "urgent": sum(r["priority"] in URGENT for r in rs), "worst_priority": worst["priority"],
            "examples": [{"review_id": r["review_id"], "app": r["app"], "evidence": r["evidence"]}
                         for r in sorted(rs, key=lambda r: (r["priority"], -SEVERITY_ORDER[r["severity"]]))[:2]],
        })

    high_priority = sorted([r for r in rows if r["priority"] in URGENT],
                           key=lambda r: (r["priority"], -SEVERITY_ORDER[r["severity"]], r["review_id"]))

    # ---- Agent 3: improvement actions grouped by action, app and dimension ---
    tickets = []
    for s in done:
        a3 = s.agent3
        if a3.ticket is None or a3.action == ActionType.NO_ACTION:
            continue
        lead = sorted(s.agent2.assessments, key=lambda a: (-SEVERITY_ORDER[a.severity.value], a.priority.value))
        lead_issue = next((i for i in s.agent1.issues if lead and i.issue_id == lead[0].issue_id), None)
        tickets.append({"action": a3.action.value, "app": s.review.app_name or s.review.app_id, "team": a3.handling_team or "—",
                        "dimension": lead_issue.dimension_name if lead_issue else "—",
                        "priority": lead[0].priority.value if lead else "—", "title": a3.ticket.title, "review_id": s.review.review_id})
    action_groups: dict[tuple, list[dict]] = defaultdict(list)
    for t in tickets:
        action_groups[(t["action"], t["app"], t["dimension"])].append(t)
    improvements = [{"action": a, "app": app, "dimension": d, "tickets": len(ts), "team": ts[0]["team"],
                     "urgent": sum(t["priority"] in URGENT for t in ts), "example_title": ts[0]["title"]}
                    for (a, app, d), ts in sorted(action_groups.items(), key=lambda kv: (-sum(t["priority"] in URGENT for t in kv[1]), -len(kv[1])))]

    # ---- Escalation (simulated) ----------------------------------------------
    candidates = [{"review_id": s.review.review_id, "app": s.review.app_name, "reference": s.agent3.escalation.complaint_reference,
                   "sla_deadline": s.agent3.escalation.sla_deadline, "days_overdue": s.agent3.escalation.sla_days_overdue}
                  for s in done if s.agent3.escalation.status == EscalationStatus.CANDIDATE_AWAITING_HUMAN_APPROVAL]
    pending = [{"review_id": s.review.review_id, "app": s.review.app_name, "missing": ", ".join(s.agent3.escalation.missing_fields)}
               for s in done if s.agent3.escalation.status == EscalationStatus.PENDING_VERIFICATION]

    # ---- Per application -------------------------------------------------------
    by_app = []
    for app, n in apps.most_common():
        rs = [r for r in rows if r["app"] == app]
        dim = Counter(r["dimension_name"] for r in rs).most_common(1)
        top = Counter(r["topic_label"] for r in rs).most_common(1)
        by_app.append({"app": app, "reviews": n, "issues": len(rs), "urgent": sum(r["priority"] in URGENT for r in rs),
                       "high_critical": sum(r["severity"] in ("High", "Critical") for r in rs),
                       "top_dimension": f"{dim[0][0]} ({dim[0][1]})" if dim else "—",
                       "top_issue": f"{top[0][0]} ({top[0][1]})" if top else "—"})
    teams = Counter(t["team"] for t in tickets)

    ins = {
        "label": PROTOTYPE_LABEL,
        "coverage": {"reviews": n_rev, "issues": n_iss, "apps": len(apps), "top_apps": apps.most_common(10),
                     "analysis_date": (analysis_date or date.today()).isoformat(),
                     "engines": sorted({s.reasoner for s in done}),
                     "synthetic_reviews": sum(s.review.is_synthetic for s in done),
                     "llm_fallback_reviews": sum(any(e.action == "LLM_FALLBACK" for e in s.decision_log) for s in done),
                     "multi_issue_reviews": sum(len(s.agent1.issues) >= 2 for s in done),
                     "needs_clarification": sum(s.agent1.is_ambiguous for s in done),
                     "revision_requests": sum(m.message_type == "REVIEW_REQUEST" for s in done for m in s.messages)},
        "feedback_types": [(t, types.get(t, 0), _pct(types.get(t, 0), n_rev)) for t in FEEDBACK_TYPES],
        "sentiment": [(t, senti.get(t, 0), _pct(senti.get(t, 0), n_rev)) for t in SENTIMENTS],
        "severity": [(t, sev.get(t, 0), _pct(sev.get(t, 0), n_iss)) for t in SEVERITIES],
        "priority": [(t, prio.get(t, 0), _pct(prio.get(t, 0), n_iss)) for t in PRIORITIES],
        "actions": [(a, c, _pct(c, n_rev)) for a, c in actions.most_common()],
        "escalation": [(e, c) for e, c in esc.most_common()],
        "escalation_candidates": candidates, "escalation_pending": pending,
        "dimensions": dimensions, "top_issues": top_issues[:10], "high_priority": high_priority,
        "improvements": improvements[:10], "tickets": len(tickets), "by_app": by_app,
        "teams": teams.most_common(),
        "actionable": sum(s.agent3.action != ActionType.NO_ACTION for s in done),
    }
    ins["recommendations"] = management_recommendations(ins)
    ins["summary"] = executive_summary(ins)
    return ins


def management_recommendations(ins: dict) -> list[dict]:
    """Action-oriented points generated by fixed rules from the counts above; each carries its basis."""
    recs = []
    cov, n_iss = ins["coverage"], ins["coverage"]["issues"]
    urgent_total = sum(c for p, c, _ in ins["priority"] if p in URGENT)
    dims = sorted(ins["dimensions"], key=lambda d: (-d["urgent"], -d["issues"]))
    if dims and dims[0]["urgent"]:
        d = dims[0]
        app_txt = f", most often in {d['top_apps'][0][0]} ({d['top_apps'][0][1]})" if d["top_apps"] else ""
        recs.append({"title": f"Prioritise {d['name']}",
                     "basis": f"{d['urgent']} of {urgent_total} P1/P2 issues ({_pct(d['urgent'], urgent_total)}%) fall in this dimension{app_txt}."})
    tech = [g for g in ins["improvements"] if g["action"] == ActionType.INTERNAL_TECHNICAL_TICKET.value]
    if tech:
        g = tech[0]
        recs.append({"title": f"Fix the {g['dimension']} problems reported in {g['app']}",
                     "basis": f"{g['tickets']} internal technical ticket(s), {g['urgent']} of them P1/P2 (Agent 3)."})
    imp = [g for g in ins["improvements"] if g["action"] == ActionType.SERVICE_IMPROVEMENT_REQUEST.value]
    if imp:
        g = imp[0]
        recs.append({"title": f"Review the service design of {g['app']} ({g['dimension']})",
                     "basis": f"{g['tickets']} service improvement request(s) generated by Agent 3."})
    info = dict((a, c) for a, c, _ in ins["actions"]).get(ActionType.REQUEST_ADDITIONAL_INFO.value, 0)
    if info:
        recs.append({"title": "Ask beneficiaries for the missing details before acting",
                     "basis": f"{info} of {cov['reviews']} reviews ({_pct(info, cov['reviews'])}%) did not name the feature, step or error "
                              "(Agent 3 action: request for additional information)."})
    if ins["escalation_candidates"] or ins["escalation_pending"]:
        recs.append({"title": "Verify previous complaints before any escalation",
                     "basis": f"{len(ins['escalation_candidates'])} simulated escalation candidate(s) await human approval; "
                              f"{len(ins['escalation_pending'])} mention a complaint that could not be verified."})
    if not recs and n_iss == 0:
        recs.append({"title": "No operational issue found in the analysed feedback",
                     "basis": f"{cov['reviews']} reviews analysed; none contained an evidence-backed service issue."})
    return recs[:5]


def executive_summary(ins: dict) -> list[str]:
    """Plain sentences built only from the computed counts."""
    cov = ins["coverage"]
    if cov["reviews"] == 0:
        return []
    out = [f"{cov['reviews']} beneficiary reviews of {cov['apps']} government application(s) were analysed by the three-agent "
           f"framework on {cov['analysis_date']}, yielding {cov['issues']} evidence-backed service issue(s)."]
    hc = sum(c for s, c, _ in ins["severity"] if s in ("High", "Critical"))
    urgent = sum(c for p, c, _ in ins["priority"] if p in URGENT)
    if cov["issues"]:
        out.append(f"{hc} issue(s) ({_pct(hc, cov['issues'])}%) were rated High or Critical because they prevented or disrupted "
                   f"service completion, and {urgent} were assigned priority P1 or P2.")
    if ins["dimensions"]:
        d = ins["dimensions"][0]
        out.append(f"The most affected service-quality dimension was {d['name']} with {d['issues']} issue(s) "
                   f"({d['share']}% of all issues).")
    neg = dict((s, c) for s, c, _ in ins["sentiment"]).get("Negative", 0)
    out.append(f"{neg} review(s) ({_pct(neg, cov['reviews'])}%) were negative in tone; tone is reported separately and never raises severity.")
    out.append(f"Agent 3 generated {ins['tickets']} simulated ticket(s); {len(ins['escalation_candidates'])} simulated escalation "
               "candidate(s) await human approval. Nothing was submitted to any government system.")
    return out
