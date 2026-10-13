"""Agent tools.

Every tool is a plain, deterministic Python function registered in
``TOOL_REGISTRY`` together with the agent(s) allowed to use it. Agents call
tools through ``BaseAgent.call_tool`` which enforces the allow-list and records
every call (reason, input, output) in the shared state.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import date, timedelta
from functools import lru_cache
from typing import Any, Callable, Optional

import config
from arabic_text import containment, find_cues, normalize, normalize_with_map, tokens, word_count
import lexicon as lx
from schemas import SEVERITY_ORDER, ComplaintRecord, RuleEvaluation


# --------------------------------------------------------------------------- #
# Knowledge-base loading
# --------------------------------------------------------------------------- #
@lru_cache(maxsize=None)
def _load_json(path: str) -> dict:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def egsqual_kb() -> dict:
    return _load_json(str(config.EGSQUAL_KB_PATH))["dimensions"]


def severity_rules_kb() -> dict:
    return _load_json(str(config.SEVERITY_RULES_PATH))


def escalation_rules_kb() -> dict:
    return _load_json(str(config.ESCALATION_RULES_PATH))


def dialect_kb() -> dict:
    return _load_json(str(config.DIALECT_LEXICON_PATH))["terms"]


# =========================================================================== #
# Agent 1 tools
# =========================================================================== #
def arabic_normalizer(text: str) -> dict:
    """Normalise Arabic orthography (hamza/taa marbuta/alef maqsura, diacritics,
    elongation) and count tokens."""
    norm = normalize(text)
    return {"normalized_text": norm, "token_count": len(tokens(text))}


def dialect_lexicon_lookup(text: str) -> dict:
    """Return Saudi-dialect terms found in the review with MSA/English glosses."""
    terms = dialect_kb()
    found = {}
    norm_cache = normalize_with_map(text)
    for term, gloss in terms.items():
        if find_cues(text, [normalize(term)], norm_cache):
            found[term] = gloss
    return {"dialect_terms": found}


def egsqual_kb_lookup(dimension_key: str) -> dict:
    """Look up an EGSQUAL dimension definition by key."""
    kb = egsqual_kb()
    key = (dimension_key or "").strip().upper()
    if key == "ALL":
        return {"found": True, "dimensions": {k: {"name": d["name_en"], "definition": d["definition"]} for k, d in kb.items()}}
    if key in kb:
        d = kb[key]
        return {"found": True, "key": key, "name": d["name_en"], "name_ar": d["name_ar"], "definition": d["definition"], "source": d["source"]}
    return {"found": False, "key": key, "valid_keys": list(kb)}


def egsqual_kb_search(text: str) -> dict:
    """Suggest the best-matching EGSQUAL dimension for a text span using the
    keywords stored in the knowledge base (used when a dimension is invalid)."""
    kb = egsqual_kb()
    scores = {}
    nc = normalize_with_map(text)
    for key, d in kb.items():
        kws = [normalize(k) for k in d.get("keywords_ar", [])]
        scores[key] = len(find_cues(text, kws, nc))
    best = max(scores, key=lambda k: scores[k])
    if scores[best] == 0:
        best = "SYSTEM_AVAILABILITY"
    return {"best_dimension": best, "scores": scores}


def evidence_verifier(review_text: str, evidence: str) -> dict:
    """Check that an evidence quote is really supported by the original review.
    Exact (normalised) substring -> score 1.0; otherwise token containment."""
    if not evidence or not evidence.strip():
        return {"verified": False, "score": 0.0, "method": "empty"}
    nr, ne = normalize(review_text), normalize(evidence)
    if ne and ne in nr:
        return {"verified": True, "score": 1.0, "method": "exact_substring"}
    score = containment(evidence, review_text)
    return {"verified": score >= 0.8, "score": round(score, 2), "method": "token_containment"}


def _clause_spans(text: str) -> list[tuple[int, int]]:
    spans, start = [], 0
    for m in re.finditer(r"[\.\!\؟\?\n،,؛;]+|\s-\s", text):
        if m.start() > start:
            spans.append((start, m.start()))
        start = m.end()
    if start < len(text):
        spans.append((start, len(text)))
    return [(s, e) for s, e in spans if text[s:e].strip()]


def evidence_locator(review_text: str, topic_key: Optional[str], current_evidence: str = "") -> dict:
    """Search the original review for the strongest span supporting an issue:
    the clause containing the topic cue, extended to an adjacent clause when
    that clause carries the failure marker."""
    topic = lx.TOPICS.get(topic_key or "", None)
    cues = topic.cues if topic else []
    clauses = _clause_spans(review_text)
    if not clauses:
        return {"found": False, "evidence": current_evidence}
    marker_cues = lx.HARD_FAILURE + lx.SOFT_PROBLEM + lx.MISSING + lx.CRASH_CUES

    def has(s, e, cl):
        return bool(find_cues(review_text[s:e], cl)) if cl else False

    # anchor clause: contains the topic cue, else contains current evidence
    anchor = None
    for i, (s, e) in enumerate(clauses):
        if (cues and has(s, e, cues)) or (current_evidence and normalize(current_evidence)[:15] in normalize(review_text[s:e])):
            anchor = i
            if has(s, e, marker_cues):
                break
    if anchor is None:
        return {"found": False, "evidence": current_evidence}
    s, e = clauses[anchor]
    if not has(s, e, marker_cues):
        for j in (anchor + 1, anchor - 1):
            if 0 <= j < len(clauses) and has(*clauses[j], marker_cues):
                s, e = min(s, clauses[j][0]), max(e, clauses[j][1])
                break
    span = review_text[s:e].strip()
    return {"found": True, "evidence": span, "contains_problem_marker": bool(find_cues(span, marker_cues))}


def ambiguity_checker(review_text: str, issues: list[dict], feedback_type: str) -> dict:
    """Rule-based check for ambiguous / incomplete feedback."""
    reasons, missing = [], []
    n_words = word_count(review_text)
    has_topic = bool(find_cues(review_text, lx.ALL_TOPIC_CUES))
    praise_only = feedback_type == "Praise" and not issues
    if not issues and not praise_only:
        reasons.append("A1: no concrete, actionable issue can be identified in the text")
        missing.append("What exactly went wrong (feature, step, error message)")
    if n_words <= 3 and not has_topic and not praise_only:
        reasons.append(f"A2: review is very short ({n_words} words) and names no feature or service")
    problems = [i for i in issues if i.get("issue_type") not in ("Suggestion", "Inquiry", "Praise")]
    if problems and all(i.get("topic") in ("GENERAL_MALFUNCTION", "x", None) and i.get("affected_service") == "unspecified" for i in problems):
        reasons.append("A3: the review reports a failure but does not state which feature/service failed")
        missing.append("Which feature or service the beneficiary was using")
    if find_cues(review_text, lx.VAGUE) and not has_topic:
        reasons.append("A4: global, unspecific complaint ('everything')")
    return {"is_ambiguous": bool(reasons), "reasons": reasons, "missing_information": sorted(set(missing))}


def service_identifier(text: str) -> dict:
    """Identify the government service / transaction mentioned in a span."""
    matches = find_cues(text, lx.SERVICE_TERMS)
    if not matches:
        return {"found": False, "affected_service": "unspecified"}
    terms = list(dict.fromkeys(m.cue for m in matches))
    return {"found": True, "affected_service": " / ".join(terms), "terms": terms,
            "matches": [(m.cue, m.start, m.end) for m in matches]}


TYPE_PRECEDENCE = ["Bug", "General Complaint", "Inquiry", "Suggestion", "Praise"]


def feedback_type_resolver(issue_types: list[str], has_praise: bool) -> dict:
    """Derive the primary feedback type from the extracted issues (issue-driven
    precedence: Bug > General Complaint > Inquiry > Suggestion > Praise)."""
    if not issue_types:
        return {"feedback_type": "Praise" if has_praise else "General Complaint", "secondary_types": []}
    primary = min(issue_types, key=TYPE_PRECEDENCE.index)
    secondary = sorted(set(issue_types) - {primary}, key=TYPE_PRECEDENCE.index)
    if has_praise and primary != "Praise":
        secondary.append("Praise")
    return {"feedback_type": primary, "secondary_types": secondary}


def issue_deduplicator(issues: list[dict]) -> dict:
    """Find issue pairs that describe the same problem (same dimension and
    overlapping evidence)."""
    dups = []
    for i in range(len(issues)):
        for j in range(i + 1, len(issues)):
            a, b = issues[i], issues[j]
            if a["dimension"] != b["dimension"]:
                continue
            na, nb = normalize(a["evidence"]), normalize(b["evidence"])
            overlap = na in nb or nb in na or containment(a["evidence"], b["evidence"]) >= 0.7
            if overlap:
                dups.append((a["issue_id"], b["issue_id"]))
    return {"duplicates": dups}


# =========================================================================== #
# Agent 2 tools
# =========================================================================== #
def impact_cue_detector(span: str) -> dict:
    """Detect operational-impact cues in a span (failure, crash, slowness,
    intermittency, workaround, recurrence, scope, risk)."""
    nc = normalize_with_map(span)

    def q(cues):
        return [span[m.start:m.end].strip() for m in find_cues(span, cues, nc)]

    risk = []
    if q(lx.TOPICS["PAYMENT"].cues):
        risk.append("FINANCIAL")
    if q(lx.TOPICS["PRIVACY"].cues):
        risk.append("PERSONAL_DATA_SECURITY")
    if q(lx.SAFETY):
        risk.append("SAFETY")
    if q(lx.LEGAL_DEADLINE):
        risk.append("LEGAL_DEADLINE")
    return {
        "hard_failure": q(lx.HARD_FAILURE),
        "crash": q(lx.CRASH_CUES),
        "freeze_or_slow": q(lx.TOPICS["STABILITY"].cues + lx.TOPICS["PERFORMANCE"].cues),
        "soft_problem": q(lx.SOFT_PROBLEM),
        "missing": q(lx.MISSING),
        "partial": q(lx.PARTIAL),
        "workaround": q(lx.WORKAROUND),
        "recurring": q(lx.RECURRING),
        "widespread": q(lx.WIDESPREAD),
        "risk_flags": risk,
    }


def sentiment_intensity(review_text: str, rating: Optional[int]) -> dict:
    """Lexicon-based sentiment intensity. Reported separately from severity."""
    nc = normalize_with_map(review_text)
    neg = len(find_cues(review_text, lx.NEGATIVE, nc))
    pos = len(find_cues(review_text, lx.PRAISE, nc))
    excl = review_text.count("!") + review_text.count("!!")
    score = (pos - neg - 0.5 * min(excl, 4)) / max(1, pos + neg + 1)
    if rating is not None:
        score = 0.5 * score + 0.5 * ((rating - 3) / 2)
    score = max(-1.0, min(1.0, score))
    if score <= -0.6:
        label = "Strongly negative"
    elif score <= -0.2:
        label = "Negative"
    elif score < 0.2:
        label = "Neutral / mixed"
    else:
        label = "Positive"
    return {"label": label, "score": round(score, 2), "negative_cues": neg, "positive_cues": pos, "rating": rating}


def severity_rule_engine(service_completion: str, scope: str, risk_flags: list[str]) -> dict:
    """Apply the PROPOSED severity rules R1-R4 (see knowledge_base/severity_priority_rules.json)."""
    serious_risk = {"FINANCIAL", "PERSONAL_DATA_SECURITY", "SAFETY"} & set(risk_flags)
    if service_completion == "BLOCKED" and (serious_risk or scope == "WIDESPREAD"):
        why = f"BLOCKED + {'risk ' + ','.join(sorted(serious_risk)) if serious_risk else 'WIDESPREAD scope'}"
        return {"severity": "Critical", "rule": "R1", "explanation": why}
    if service_completion == "BLOCKED":
        return {"severity": "High", "rule": "R2", "explanation": "Service completion BLOCKED, no qualifying risk/scope"}
    if service_completion == "PARTIALLY_BLOCKED" and risk_flags:
        return {"severity": "High", "rule": "R2", "explanation": f"PARTIALLY_BLOCKED + risk {','.join(risk_flags)}"}
    if service_completion == "PARTIALLY_BLOCKED":
        return {"severity": "Medium", "rule": "R3", "explanation": "PARTIALLY_BLOCKED (fails sometimes / workaround exists)"}
    if service_completion == "DEGRADED" and (scope in ("RECURRING", "WIDESPREAD") or risk_flags):
        return {"severity": "Medium", "rule": "R3", "explanation": f"DEGRADED + {scope if scope != 'INDIVIDUAL' else 'risk'}"}
    return {"severity": "Low", "rule": "R4", "explanation": f"{service_completion}, individual scope, no risk"}


_PRIORITY_LEVELS = ["P1", "P2", "P3", "P4"]


def priority_rule_engine(severity: str, scope: str, evidence_insufficient: bool = False) -> dict:
    """Apply the PROPOSED priority rules P-BASE, P-SCOPE, P-VERIFY."""
    kb = severity_rules_kb()
    pr = kb["priority_map"][severity]
    rules = ["P-BASE"]
    if scope == "WIDESPREAD" and SEVERITY_ORDER[severity] >= SEVERITY_ORDER["Medium"]:
        pr = _PRIORITY_LEVELS[max(0, _PRIORITY_LEVELS.index(pr) - 1)]
        rules.append("P-SCOPE")
    if evidence_insufficient:
        cap = kb.get("insufficient_evidence_priority_cap", "P3")
        if _PRIORITY_LEVELS.index(pr) < _PRIORITY_LEVELS.index(cap):
            pr = cap
            rules.append("P-VERIFY")
    return {"priority": pr, "rules": rules}


# =========================================================================== #
# Agent 3 tools
# =========================================================================== #
_REF_RE = re.compile(r"(SYN-\d{4}-\d{3,6}|[A-Z]{2,5}-\d{3,10}|(?<!\d)\d{6,12}(?!\d))")


def complaint_reference_extractor(review_text: str) -> dict:
    """Detect whether the review mentions a previous complaint and extract a
    reference number if one is written. Never invents a reference."""
    mentions = [review_text[m.start:m.end] for m in find_cues(review_text, lx.COMPLAINT_MENTION)]
    norm_digits = normalize_with_map(review_text)[0].upper()
    refs = _REF_RE.findall(norm_digits)
    # a bare number only counts as a complaint reference if a complaint is mentioned
    refs = [r for r in refs if not r.isdigit() or mentions]
    return {"complaint_mentioned": bool(mentions or refs), "mentions": mentions, "reference": refs[0] if refs else None}


def complaint_history_lookup(reference: Optional[str], app_id: Optional[str] = None) -> dict:
    """Look up a complaint in the (SYNTHETIC) complaint-history store."""
    if not reference:
        return {"found": False, "reason": "no complaint reference available"}
    data = _load_json(str(config.COMPLAINT_HISTORY_PATH))["records"]
    for rec in data:
        if rec["complaint_reference"].upper() == reference.upper():
            return {"found": True, "record": rec}
    return {"found": False, "reason": f"reference {reference} not found in complaint history"}


def sla_checker(record: dict, reference_date: Optional[date] = None) -> dict:
    """Compute the SLA response deadline and whether it has passed."""
    rules = escalation_rules_kb()
    today = reference_date or config.SLA_REFERENCE_DATE or date.today()
    if not record.get("submission_date"):
        return {"available": False, "reason": "submission date missing"}
    sla_days = record.get("sla_response_days")
    sla_source = "complaint record"
    if sla_days is None:
        sla_days = rules["default_response_sla_days"]
        sla_source = "configured default (placeholder)"
    submitted = date.fromisoformat(record["submission_date"])
    deadline = submitted + timedelta(days=int(sla_days))
    overdue = (today - deadline).days
    return {
        "available": True,
        "reference_date": today.isoformat(),
        "deadline": deadline.isoformat(),
        "sla_days": sla_days,
        "sla_source": sla_source,
        "exceeded": overdue > 0,
        "days_overdue": max(0, overdue),
    }


def escalation_rule_engine(
    severity: Optional[str],
    complaint_record: Optional[dict],
    sla: Optional[dict],
    complaint_mentioned: bool,
) -> dict:
    """Evaluate the PROPOSED escalation-eligibility rules E1-E5."""
    kb = escalation_rules_kb()
    desc = {r["id"]: r["description"] for r in kb["rules"]}
    min_sev = kb["min_severity_for_escalation"]
    evals: list[RuleEvaluation] = []
    missing: list[str] = []

    # E1 - previous complaint exists
    if complaint_record:
        evals.append(RuleEvaluation(rule_id="E1", description=desc["E1"], result="MET", evidence=f"Record {complaint_record['complaint_reference']} found"))
    elif complaint_mentioned:
        evals.append(RuleEvaluation(rule_id="E1", description=desc["E1"], result="UNKNOWN", evidence="Review mentions a previous complaint but no verifiable reference/record"))
        missing += ["complaint_reference", "complaint_status", "submission_date"]
    else:
        evals.append(RuleEvaluation(rule_id="E1", description=desc["E1"], result="NOT_MET", evidence="No previous complaint mentioned"))

    # E2 - unresolved
    if complaint_record:
        st = complaint_record.get("complaint_status", "")
        res = "MET" if st in kb["unresolved_statuses"] else "NOT_MET"
        evals.append(RuleEvaluation(rule_id="E2", description=desc["E2"], result=res, evidence=f"Status: {st}"))
    else:
        evals.append(RuleEvaluation(rule_id="E2", description=desc["E2"], result="UNKNOWN", evidence="Status not available"))

    # E3 - SLA passed
    if sla and sla.get("available"):
        evals.append(RuleEvaluation(rule_id="E3", description=desc["E3"], result="MET" if sla["exceeded"] else "NOT_MET",
                                    evidence=f"Deadline {sla['deadline']} ({sla['sla_days']} days, {sla['sla_source']}); reference date {sla['reference_date']}"))
    else:
        evals.append(RuleEvaluation(rule_id="E3", description=desc["E3"], result="UNKNOWN", evidence="SLA data not available"))
        if complaint_mentioned or complaint_record:
            missing.append("sla_response_days")

    # E4 - severity threshold
    sev_ok = severity is not None and SEVERITY_ORDER[severity] >= SEVERITY_ORDER[min_sev]
    evals.append(RuleEvaluation(rule_id="E4", description=desc["E4"], result="MET" if sev_ok else "NOT_MET", evidence=f"Severity {severity} vs minimum {min_sev}"))

    # E5 - human approval (always pending in the prototype)
    evals.append(RuleEvaluation(rule_id="E5", description=desc["E5"], result="UNKNOWN", evidence="Awaiting human officer approval (mandatory)"))

    r = {e.rule_id: e.result for e in evals}
    if all(r[k] == "MET" for k in ("E1", "E2", "E3", "E4")):
        status = "CANDIDATE_AWAITING_HUMAN_APPROVAL"
    elif r["E4"] == "MET" and any(r[k] == "UNKNOWN" for k in ("E1", "E2", "E3")) and (complaint_mentioned or complaint_record):
        status = "PENDING_VERIFICATION"
    elif not complaint_mentioned and not complaint_record:
        status = "NOT_ELIGIBLE"
    else:
        status = "NOT_ELIGIBLE"
    return {"status": status, "rules": [e.model_dump() for e in evals], "missing_fields": sorted(set(missing))}


REQUIRED_TICKET_FIELDS = [
    "ticket_id", "title", "application", "affected_service", "issue_description", "severity",
    "priority", "recommended_action", "justification",
]


def ticket_fields_validator(ticket: dict) -> dict:
    """Check that a simulated ticket contains every required field."""
    missing = [f for f in REQUIRED_TICKET_FIELDS if not str(ticket.get(f, "")).strip()]
    return {"complete": not missing, "missing_fields": missing}


# =========================================================================== #
# Registry (agent-specific tool allow-lists)
# =========================================================================== #
@dataclass
class ToolSpec:
    name: str
    func: Callable[..., Any]
    agents: tuple[str, ...]
    description: str


TOOL_REGISTRY: dict[str, ToolSpec] = {
    s.name: s
    for s in [
        ToolSpec("arabic_normalizer", arabic_normalizer, ("Agent1",), "Normalise Arabic orthography"),
        ToolSpec("dialect_lexicon_lookup", dialect_lexicon_lookup, ("Agent1",), "Saudi dialect glossary lookup"),
        ToolSpec("egsqual_kb_lookup", egsqual_kb_lookup, ("Agent1",), "EGSQUAL dimension definition lookup"),
        ToolSpec("egsqual_kb_search", egsqual_kb_search, ("Agent1",), "Suggest EGSQUAL dimension by keywords"),
        ToolSpec("evidence_verifier", evidence_verifier, ("Agent1", "Agent2"), "Verify evidence is in the review"),
        ToolSpec("evidence_locator", evidence_locator, ("Agent1",), "Find stronger evidence span"),
        ToolSpec("ambiguity_checker", ambiguity_checker, ("Agent1",), "Detect ambiguous / incomplete feedback"),
        ToolSpec("service_identifier", service_identifier, ("Agent1", "Agent3"), "Identify affected service"),
        ToolSpec("feedback_type_resolver", feedback_type_resolver, ("Agent1",), "Issue-driven feedback type"),
        ToolSpec("issue_deduplicator", issue_deduplicator, ("Agent1", "Agent2"), "Detect duplicate issues"),
        ToolSpec("impact_cue_detector", impact_cue_detector, ("Agent2",), "Detect operational impact cues"),
        ToolSpec("sentiment_intensity", sentiment_intensity, ("Agent2",), "Sentiment intensity (not severity)"),
        ToolSpec("severity_rule_engine", severity_rule_engine, ("Agent2",), "Apply proposed severity rules"),
        ToolSpec("priority_rule_engine", priority_rule_engine, ("Agent2",), "Apply proposed priority rules"),
        ToolSpec("complaint_reference_extractor", complaint_reference_extractor, ("Agent3",), "Detect prior complaint reference"),
        ToolSpec("complaint_history_lookup", complaint_history_lookup, ("Agent3",), "Look up complaint history (synthetic)"),
        ToolSpec("sla_checker", sla_checker, ("Agent3",), "Check SLA response deadline"),
        ToolSpec("escalation_rule_engine", escalation_rule_engine, ("Agent3",), "Apply proposed escalation rules"),
        ToolSpec("ticket_fields_validator", ticket_fields_validator, ("Agent3",), "Validate simulated ticket fields"),
    ]
}


def complaint_record_model(rec: Optional[dict]) -> Optional[ComplaintRecord]:
    return ComplaintRecord(**rec) if rec else None
