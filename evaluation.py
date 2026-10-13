"""Scenario-based evaluation.

Two kinds of evaluation:

1. Behavioural checks (no labels needed) - they check the framework's own
   behaviour against its rules and the available evidence:
   evidence support, multi-issue detection, ambiguity flagging, severity
   rule compliance, escalation consistency, ticket completeness, and whether
   inter-agent review/revision actually occurred.

2. Optional Precision / Recall / F1 - computed ONLY when the user uploads
   verified reference labels. Nothing is computed (and no score is shown)
   without them. No scores are fabricated.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pandas as pd

import tools
from schemas import SEVERITY_ORDER, ActionType, EscalationStatus, VerificationStatus, WorkflowState

_PRIO = ["P1", "P2", "P3", "P4"]


@dataclass
class Check:
    check: str
    passed: Optional[bool]  # None = not applicable
    detail: str


def behavioural_checks(state: WorkflowState, expect: dict | None = None) -> list[Check]:
    expect = expect or {}
    a1, a2, a3, text = state.agent1, state.agent2, state.agent3, state.review.text
    out: list[Check] = []
    if a1 is None or a2 is None or a3 is None:
        return [Check("workflow_completed", False, f"Workflow incomplete: {state.errors}")]
    out.append(Check("workflow_completed", True, " -> ".join(state.node_trace)))

    # C1 evidence supported
    if a1.issues:
        bad = [i.issue_id for i in a1.issues if not tools.evidence_verifier(text, i.evidence)["verified"]]
        out.append(Check("issues_supported_by_review", not bad, "all evidence quotes found in the review" if not bad else f"unsupported: {bad}"))
    else:
        out.append(Check("issues_supported_by_review", None, "no issues extracted"))

    # C2 multiple issues
    if "min_issues" in expect:
        n = len(a1.issues)
        out.append(Check("issue_count", n >= expect["min_issues"], f"{n} issue(s) extracted (expected >= {expect['min_issues']})"))

    # C3 ambiguity flagged
    if "ambiguous" in expect:
        out.append(Check("ambiguity_flag", a1.is_ambiguous == expect["ambiguous"],
                         f"is_ambiguous={a1.is_ambiguous} (expected {expect['ambiguous']}); reasons={a1.ambiguity_reasons}"))

    if "feedback_type" in expect:
        out.append(Check("feedback_type", a1.feedback_type == expect["feedback_type"], f"{a1.feedback_type} (expected {expect['feedback_type']})"))
    if "issue_types" in expect:
        got = sorted({i.issue_type for i in a1.issues})
        out.append(Check("issue_types", set(expect["issue_types"]) <= set(got), f"{got} (expected to include {expect['issue_types']})"))

    # C4 severity/priority follow the proposed rules
    if a2.assessments:
        mismatches = []
        for a in a2.assessments:
            sev = tools.severity_rule_engine(a.service_completion, a.scope, a.risk_flags)["severity"]
            pr = tools.priority_rule_engine(sev, a.scope, a.verification_status == VerificationStatus.INSUFFICIENT_EVIDENCE)["priority"]
            if sev != a.severity.value or pr != a.priority.value:
                mismatches.append(f"{a.issue_id}: got {a.severity.value}/{a.priority.value}, rules give {sev}/{pr}")
        out.append(Check("severity_priority_follow_rules", not mismatches, "all consistent with R1-R4 / P-rules" if not mismatches else "; ".join(mismatches)))
    else:
        out.append(Check("severity_priority_follow_rules", None, "no assessments (no issues)"))
    if expect.get("no_severity"):
        out.append(Check("no_severity_without_issue", a2.overall_severity is None, f"overall severity = {a2.overall_severity}"))
    if "min_severity" in expect:
        sev = a2.overall_severity.value if a2.overall_severity else None
        ok = sev is not None and SEVERITY_ORDER[sev] >= SEVERITY_ORDER[expect["min_severity"]]
        out.append(Check("severity_at_least", ok, f"{sev} (expected >= {expect['min_severity']})"))
    if "max_severity" in expect:
        sev = a2.overall_severity.value if a2.overall_severity else None
        ok = sev is None or SEVERITY_ORDER[sev] <= SEVERITY_ORDER[expect["max_severity"]]
        out.append(Check("severity_at_most", ok, f"{sev} (expected <= {expect['max_severity']})"))
    if "max_priority" in expect:
        pr = a2.overall_priority.value if a2.overall_priority else None
        ok = pr is not None and _PRIO.index(pr) <= _PRIO.index(expect["max_priority"])
        out.append(Check("priority_at_least", ok, f"{pr} (expected {expect['max_priority']} or more urgent)"))

    # C5 escalation consistent with evidence
    esc = a3.escalation
    problems = []
    if esc.status == EscalationStatus.CANDIDATE_AWAITING_HUMAN_APPROVAL:
        if esc.complaint_record is None:
            problems.append("candidate without a complaint record")
        if any(r.result != "MET" for r in esc.rules if r.rule_id in ("E1", "E2", "E3", "E4")):
            problems.append("candidate although E1-E4 not all MET")
        if not esc.human_approval_required:
            problems.append("human approval not required")
    if esc.status == EscalationStatus.PENDING_VERIFICATION and not esc.missing_fields:
        problems.append("pending verification without listing missing fields")
    if esc.complaint_record is not None and esc.complaint_reference is None:
        problems.append("complaint record without a reference")
    if esc.complaint_reference and esc.complaint_reference not in text.upper():
        problems.append("complaint reference not present in the review (invented?)")
    if a3.ticket is not None and not a3.ticket.simulated:
        problems.append("ticket not marked simulated")
    out.append(Check("escalation_follows_evidence", not problems, f"status={esc.status.value}" if not problems else "; ".join(problems)))
    if "escalation" in expect:
        out.append(Check("escalation_status", esc.status.value == expect["escalation"], f"{esc.status.value} (expected {expect['escalation']})"))

    # C6 ticket fields
    if a3.action == ActionType.NO_ACTION:
        out.append(Check("ticket_required_fields", a3.ticket is None, "no ticket for 'No action required'"))
    else:
        v = tools.ticket_fields_validator(a3.ticket.model_dump()) if a3.ticket else {"complete": False, "missing_fields": ["ticket"]}
        out.append(Check("ticket_required_fields", v["complete"], "all required fields present" if v["complete"] else f"missing {v['missing_fields']}"))

    if "actions" in expect:
        out.append(Check("action", a3.action.value in expect["actions"], f"{a3.action.value} (expected one of {expect['actions']})"))

    # C7 inter-agent review / revision
    req = [m for m in state.messages if m.message_type == "REVIEW_REQUEST"]
    resp = [m for m in state.messages if m.message_type == "REVIEW_RESPONSE"]
    detail = f"{len(req)} review request(s), {len(resp)} response(s), {state.revision_count} round(s) (max {state.max_revisions})"
    if expect.get("revision"):
        out.append(Check("inter_agent_revision", bool(req) and bool(resp) and state.revision_count >= 1, detail))
    else:
        out.append(Check("inter_agent_revision", None, detail + " (informational)"))
    out.append(Check("revision_loop_bounded", state.revision_count <= state.max_revisions, detail))
    return out


def checks_table(results: list[tuple[str, WorkflowState, dict]]) -> pd.DataFrame:
    rows = []
    for key, state, expect in results:
        for c in behavioural_checks(state, expect):
            rows.append({"scenario": key, "check": c.check,
                         "result": "N/A" if c.passed is None else ("PASS" if c.passed else "FAIL"), "detail": c.detail})
    return pd.DataFrame(rows)


def summary(df: pd.DataFrame) -> dict:
    counts = df["result"].value_counts().to_dict()
    applicable = counts.get("PASS", 0) + counts.get("FAIL", 0)
    return {"checks_passed": counts.get("PASS", 0), "checks_failed": counts.get("FAIL", 0),
            "not_applicable": counts.get("N/A", 0), "applicable": applicable}


# --------------------------------------------------------------------------- #
# Optional P/R/F1 with VERIFIED reference labels only
# --------------------------------------------------------------------------- #
LABEL_FIELDS = {
    "feedback_type": lambda s: s.agent1.feedback_type if s.agent1 else None,
    "severity": lambda s: s.agent2.overall_severity.value if s.agent2 and s.agent2.overall_severity else "None",
    "priority": lambda s: s.agent2.overall_priority.value if s.agent2 and s.agent2.overall_priority else "None",
    "action": lambda s: s.agent3.action.value if s.agent3 else None,
    "escalation_status": lambda s: s.agent3.escalation.status.value if s.agent3 else None,
    "dimensions": lambda s: ";".join(sorted({i.dimension for i in s.agent1.issues})) if s.agent1 else "",
}


def prf_from_reference(reference: pd.DataFrame, states: list[WorkflowState]) -> dict[str, pd.DataFrame]:
    """Compute per-class Precision/Recall/F1 for each label column present in
    `reference` (must contain `review_id`). `dimensions` is multi-label
    (semicolon-separated) and uses set-based micro counts."""
    if reference is None or reference.empty or "review_id" not in reference.columns:
        raise ValueError("Reference labels must be a CSV with a 'review_id' column plus label columns")
    pred = {s.review.review_id: s for s in states if s.agent3}
    ref = reference[reference["review_id"].astype(str).isin(pred)]
    results = {}
    for field, getter in LABEL_FIELDS.items():
        if field not in ref.columns:
            continue
        y_true, y_pred = [], []
        for _, row in ref.iterrows():
            if pd.isna(row[field]):
                continue
            y_true.append(str(row[field]).strip())
            y_pred.append(str(getter(pred[str(row["review_id"])])))
        if not y_true:
            continue
        results[field] = _multilabel_prf(y_true, y_pred) if field == "dimensions" else _multiclass_prf(y_true, y_pred)
    return results


def _multiclass_prf(y_true: list[str], y_pred: list[str]) -> pd.DataFrame:
    labels = sorted(set(y_true) | set(y_pred))
    rows = []
    for lab in labels:
        tp = sum(t == lab and p == lab for t, p in zip(y_true, y_pred))
        fp = sum(t != lab and p == lab for t, p in zip(y_true, y_pred))
        fn = sum(t == lab and p != lab for t, p in zip(y_true, y_pred))
        rows.append(_row(lab, tp, fp, fn, support=sum(t == lab for t in y_true)))
    df = pd.DataFrame(rows)
    macro = {"label": "MACRO AVG", "precision": df["precision"].mean(), "recall": df["recall"].mean(), "f1": df["f1"].mean(), "support": len(y_true)}
    acc = {"label": "ACCURACY", "precision": None, "recall": None, "f1": sum(t == p for t, p in zip(y_true, y_pred)) / len(y_true), "support": len(y_true)}
    return pd.concat([df, pd.DataFrame([macro, acc])], ignore_index=True)


def _multilabel_prf(y_true: list[str], y_pred: list[str]) -> pd.DataFrame:
    split = lambda s: {x.strip() for x in str(s).split(";") if x.strip() and x.strip() != "nan"}  # noqa: E731
    T, P = [split(t) for t in y_true], [split(p) for p in y_pred]
    labels = sorted(set().union(*T, *P))
    rows = []
    for lab in labels:
        tp = sum(lab in t and lab in p for t, p in zip(T, P))
        fp = sum(lab not in t and lab in p for t, p in zip(T, P))
        fn = sum(lab in t and lab not in p for t, p in zip(T, P))
        rows.append(_row(lab, tp, fp, fn, support=sum(lab in t for t in T)))
    tp = sum(len(t & p) for t, p in zip(T, P))
    fp = sum(len(p - t) for t, p in zip(T, P))
    fn = sum(len(t - p) for t, p in zip(T, P))
    rows.append(_row("MICRO AVG", tp, fp, fn, support=sum(len(t) for t in T)))
    return pd.DataFrame(rows)


def _row(label, tp, fp, fn, support):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return {"label": label, "precision": round(p, 3), "recall": round(r, 3), "f1": round(f, 3), "support": support}
