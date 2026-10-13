"""Offline, transparent rule-based reasoner.

It produces exactly the same structured outputs (``LLMAgent1Analysis``,
``LLMImpactAssessment``) that the LLM is asked to produce. It is used when no
API key is configured, when an LLM call fails, and in the automated tests.

It is a deliberately simple lexicon + clause-segmentation approach for Arabic
(MSA and Saudi dialect); the *agentic* behaviour (tool use, verification,
revision, escalation decisions) lives in the agents, not here.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import lexicon as lx
import tools
from arabic_text import find_cues, normalize, normalize_with_map, tokens
from schemas import LLMAgent1Analysis, LLMImpact, LLMImpactAssessment, LLMIssue, ReviewInput

_TECH_DIMS = {"SYSTEM_AVAILABILITY"}
_CUE_TO_TOPIC = {}
for _t in lx.TOPICS.values():
    for _c in _t.cues:
        _CUE_TO_TOPIC.setdefault(_c, _t.key)
_SPLIT_CUES = lx.ALL_TOPIC_CUES + lx.HARD_FAILURE + lx.SOFT_PROBLEM + lx.MISSING + lx.SUGGESTION
_CONTRAST = set(normalize(w) for w in ["لكن", "ولكن", "بس", "الا ان", "غير ان"])


@dataclass
class Piece:
    start: int
    end: int
    clause: int
    contrast_before: bool = False
    contrast_inside: bool = False
    question: bool = False
    feats: dict = field(default_factory=dict)


def topics_in(text: str, nc=None) -> list[str]:
    seen = []
    for m in find_cues(text, lx.ALL_TOPIC_CUES, nc):
        k = _CUE_TO_TOPIC[m.cue]
        if k not in seen:
            seen.append(k)
    return seen


def _praise_hits(text: str, nc) -> int:
    norm = nc[0]
    count = 0
    for m in find_cues(text, lx.PRAISE, nc):
        before = norm[: m.norm_start].split()
        if before and before[-1] in lx.NEGATION:
            continue
        count += 1
    return count


def _suggestion_hits(text: str, nc) -> int:
    norm = nc[0]
    count = 0
    for m in find_cues(text, lx.SUGGESTION, nc):
        before = norm[: m.norm_start].split()
        if before and before[-1] == "علي":  # "مشكورين على تسهيل" = thanks for making it easy
            continue
        count += 1
    return count


def features(text: str) -> dict:
    nc = normalize_with_map(text)
    return {
        "topics": topics_in(text, nc),
        "hard": bool(find_cues(text, lx.HARD_FAILURE, nc)),
        "soft": bool(find_cues(text, lx.SOFT_PROBLEM, nc)),
        "error": bool(find_cues(text, lx.ERROR_CUES, nc)),
        "missing": bool(find_cues(text, lx.MISSING, nc)),
        "sugg": _suggestion_hits(text, nc) > 0,
        "inquiry": bool(find_cues(text, [c for c in lx.INQUIRY if c not in ("؟", "?")], nc, whole_word=True)),
        "praise": _praise_hits(text, nc),
        "negative": len(find_cues(text, lx.NEGATIVE, nc)),
        "crash": bool(find_cues(text, lx.CRASH_CUES, nc)),
        "service": bool(find_cues(text, lx.SERVICE_TERMS, nc)),
        "contrast": bool(find_cues(text, [normalize(w) for w in ("بس", "لكن", "ولكن")], nc, whole_word=True)),
        "done": bool(find_cues(text, [normalize(w) for w in ("تم الحجز", "تم الطلب", "تم السداد", "تم التسجيل", "طلع لي تم")], nc)),
        "words": len(tokens(text)),
    }


def _inherent(topics: list[str]) -> bool:
    return any(lx.TOPICS[t].inherent_problem for t in topics)


def _has_problem(f: dict) -> bool:
    return f["hard"] or f["soft"] or f["missing"] or _inherent(f["topics"])


# --------------------------------------------------------------------------- #
# Segmentation
# --------------------------------------------------------------------------- #
def segment(text: str) -> list[Piece]:
    pieces: list[Piece] = []
    clauses = tools._clause_spans(text)
    for ci, (cs, ce) in enumerate(clauses):
        tail = text[ce: ce + 2]
        question = "؟" in tail or "?" in tail
        # 1) split on connector words (whole words only)
        sub = text[cs:ce]
        cuts = []
        for m in find_cues(sub, lx.CONNECTORS, whole_word=True):
            cuts.append((cs + m.start, cs + m.end, normalize(sub[m.start:m.end]).lstrip("و") in _CONTRAST or normalize(sub[m.start:m.end]) in _CONTRAST))
        for m in re.finditer(r"(?<=\s)و(?=\s)", sub):  # standalone conjunction " و "
            cuts.append((cs + m.start(), cs + m.end(), False))
        cuts.sort()
        bounds = []
        pos = cs
        contrast = False
        for s, e, is_contrast in cuts:
            if s > pos:
                bounds.append((pos, s, contrast))
            pos, contrast = e, is_contrast
        bounds.append((pos, ce, contrast))
        for (ps, pe, contrast) in bounds:
            if not text[ps:pe].strip():
                continue
            # 2) split before "و"-prefixed cue words (attached conjunction)
            seg_text = text[ps:pe]
            split_points = [ps + m.start for m in find_cues(seg_text, _SPLIT_CUES)
                            if seg_text[m.start] == "و" and m.start > 0]
            pts = [ps] + sorted(set(split_points)) + [pe]
            for k in range(len(pts) - 1):
                a, b = pts[k], pts[k + 1]
                if text[a:b].strip():
                    pieces.append(Piece(a, b, ci, contrast_before=(contrast and k == 0), question=question))
    # 3) split pieces that still contain two different topics at a "و" word
    refined: list[Piece] = []
    for p in pieces:
        refined.extend(_split_multi_topic(text, p))
    for p in refined:
        p.feats = features(text[p.start:p.end])
    return refined


def _split_multi_topic(text: str, p: Piece) -> list[Piece]:
    seg = text[p.start:p.end]
    if len(topics_in(seg)) < 2:
        return [p]
    for m in re.finditer(r"(?<=\s)و\S+", seg):
        left, right = seg[: m.start()], seg[m.start():]
        tl, tr = topics_in(left), topics_in(right)
        if tl and tr and set(tl) != set(tr) and len(tokens(left)) >= 2:
            first = Piece(p.start, p.start + m.start(), p.clause, p.contrast_before, question=p.question)
            second = Piece(p.start + m.start(), p.end, p.clause, question=p.question)
            return [first] + _split_multi_topic(text, second)
    return [p]


def _merge(text: str, a: Piece, b: Piece) -> Piece:
    m = Piece(min(a.start, b.start), max(a.end, b.end), a.clause, a.contrast_before,
              contrast_inside=a.contrast_inside or b.contrast_inside or b.contrast_before,
              question=a.question or b.question)
    m.feats = features(text[m.start:m.end])
    return m


def merge_pieces(text: str, pieces: list[Piece]) -> list[Piece]:
    # Pass A: marker-only piece (no topic, has problem marker) -> merge into previous piece of same clause
    out: list[Piece] = []
    for p in pieces:
        f = p.feats
        if out and out[-1].clause == p.clause and not f["topics"] and (f["hard"] or f["soft"] or f["missing"]) and not f["sugg"]:
            out[-1] = _merge(text, out[-1], p)
        else:
            out.append(p)
    # Pass A2: a short suggestion-marker fragment -> merge with the following piece of the same clause
    merged: list[Piece] = []
    i = 0
    while i < len(out):
        p = out[i]
        if (p.feats["sugg"] and p.feats["words"] < 3 and not p.feats["topics"]
                and i + 1 < len(out) and out[i + 1].clause == p.clause):
            out[i + 1] = _merge(text, p, out[i + 1])
        else:
            merged.append(p)
        i += 1
    out = merged
    # Pass B: topic-only piece (no problem / suggestion / inquiry) -> merge into following piece of same clause with a problem
    res: list[Piece] = []
    i = 0
    while i < len(out):
        p = out[i]
        f = p.feats
        if (i + 1 < len(out) and out[i + 1].clause == p.clause and f["topics"] and not _has_problem(f)
                and not f["sugg"] and not f["inquiry"] and not f["praise"] and _has_problem(out[i + 1].feats)):
            out[i + 1] = _merge(text, p, out[i + 1])
        else:
            res.append(p)
        i += 1
    # Pass C: consecutive pieces sharing the same primary topic (across clauses) -> merge
    final: list[Piece] = []
    for p in res:
        if final:
            q = final[-1]
            tp, tq = _primary_topic(p.feats["topics"]), _primary_topic(q.feats["topics"])
            same_kind = q.feats["sugg"] == p.feats["sugg"]
            if tp and tp == tq and same_kind and not q.feats["praise"] and p.start - q.end < 40:
                final[-1] = _merge(text, q, p)
                continue
        final.append(p)
    return final


def _primary_topic(topics: list[str]) -> str | None:
    if not topics:
        return None
    return max(topics, key=lambda k: lx.TOPICS[k].specificity)


# --------------------------------------------------------------------------- #
# Agent 1 offline reasoning
# --------------------------------------------------------------------------- #
def _issue_type(f: dict, topic: str, question: bool, rhetorical_context: bool) -> str:
    t = lx.TOPICS.get(topic)
    technical = bool(t and (t.gateway or t.problem_dimension in _TECH_DIMS))
    if f["sugg"] and not f["hard"] and not f["error"]:
        return "Suggestion"
    # a question word inside a negative/contrastive complaint is usually rhetorical
    is_question = question or (f["inquiry"] and not rhetorical_context)
    if is_question and not f["hard"] and not f["soft"] and not f["missing"]:
        return "Inquiry"
    if f["missing"] and f.get("service") and (f.get("contrast") or f.get("done")):
        return "Bug"
    if f["hard"] and (technical or topic == "GENERAL_MALFUNCTION" or topic in ("REPORTING", "LANGUAGE", "DATA_ACCURACY")):
        return "Bug"
    inherent_tech = {"STABILITY", "PERFORMANCE"} & set(f["topics"])
    if topic in ("STABILITY", "PERFORMANCE") or inherent_tech or f["crash"] or f["error"] or (f["soft"] and technical):
        return "Bug"
    return "General Complaint"


def _affected_service(evidence: str, topic: str, tdef) -> str:
    """Topic label, refined with service terms that are not part of the topic cue itself."""
    svc = tools.service_identifier(evidence)
    topic_spans = [(m.start, m.end) for m in find_cues(evidence, tdef.cues)] if tdef.cues else []
    extra = []
    if svc["found"]:
        for term, s, e in svc["matches"]:
            if not any(s < te and e > ts for ts, te in topic_spans):
                extra.append(term)
    extra = list(dict.fromkeys(extra))
    if topic in ("GENERAL_MALFUNCTION",):
        return " / ".join(extra) if extra else "unspecified"
    return tdef.label_en + (f" ({' / '.join(extra)})" if extra else "")


def _description(itype: str, topic: str, f: dict) -> str:
    t = lx.TOPICS.get(topic, lx.GENERIC_TOPIC)
    label = t.label_en
    if itype == "Suggestion":
        return f"Improvement suggestion - {label.lower()}" if topic != "GENERAL_MALFUNCTION" else "Feature / service-option request"
    if itype == "Inquiry":
        return "Beneficiary question about a service / feature" if topic == "GENERAL_MALFUNCTION" else f"Beneficiary question about {label.lower()}"
    if topic == "STABILITY":
        return "App closes unexpectedly / crashes" if f["crash"] else "App freezes / hangs"
    if topic == "PERFORMANCE":
        return "App is slow / takes too long"
    if topic == "SUPPORT_RESPONSE":
        return "Complaint handling: no response / no action from the service provider"
    if itype == "Bug" and f["missing"] and f.get("service") and not f["hard"]:
        return f"{label if topic != 'GENERAL_MALFUNCTION' else 'Service transaction'}: reported as done but not fulfilled"
    if topic == "GENERAL_MALFUNCTION" and f["missing"] and not f["hard"]:
        return "Expected feature / option is missing"
    if topic == "GENERAL_MALFUNCTION" and f["error"] and f.get("service"):
        return "Service transaction fails with an error"
    if topic == "GENERAL_MALFUNCTION" and f["error"] and not f["hard"]:
        return "Errors / technical faults reported (feature not specified)"
    if topic == "GENERAL_MALFUNCTION":
        return "App or service reported as not working (feature not specified)"
    if itype == "Bug":
        if f["crash"]:
            return f"{label}: app crashes / exits during this step"
        if "STABILITY" in f["topics"]:
            return f"{label}: app freezes during this step"
        if "PERFORMANCE" in f["topics"]:
            return f"{label}: slow during this step"
        if f["hard"]:
            return f"{label}: failure reported (step does not work as expected)"
        return f"{label}: malfunction / error reported"
    if f["missing"]:
        return "Expected feature / option is missing" if topic == "GENERAL_MALFUNCTION" else f"{label}: expected feature or channel is missing"
    return f"Dissatisfaction with {label.lower()}"


def _missing_info(itype: str, review_text: str, affected_service: str) -> list[str]:
    if itype not in ("Bug", "General Complaint"):
        return []
    miss = []
    if not find_cues(review_text, lx.ERROR_MESSAGE_TERMS):
        miss.append("Exact error message shown (if any)")
    if not find_cues(review_text, lx.DEVICE_TERMS):
        miss.append("Device / OS and app version")
    if affected_service == "unspecified":
        miss.append("Which service / transaction was being performed")
    return miss


def offline_agent1_analysis(review: ReviewInput) -> LLMAgent1Analysis:
    text = review.text
    pieces = merge_pieces(text, segment(text))
    neg_context = (review.rating is not None and review.rating <= 2)
    issues: list[LLMIssue] = []
    issue_start: list[int] = []  # start offset of each issue's evidence span
    last_issue_piece = -2
    praise_total = 0
    for idx, p in enumerate(pieces):
        f = p.feats
        praise_total += f["praise"]
        topic = _primary_topic(f["topics"])
        problem = _has_problem(f) or p.contrast_inside
        skip = False
        if topic is None:
            if f["hard"] or f["error"]:
                topic = "GENERAL_MALFUNCTION"
            elif (f["sugg"] or f["missing"]) and f["words"] >= 3 and not f["praise"]:
                topic = "GENERAL_MALFUNCTION"
            elif (p.question or f["inquiry"]) and f["words"] >= 3 and not f["praise"] and not f["negative"] and not neg_context:
                topic = "GENERAL_MALFUNCTION"  # plain question about an unnamed feature/service
            else:
                skip = True
        else:
            if not (problem or f["sugg"] or f["inquiry"] or p.question):
                if not (f["negative"] or (neg_context and p.contrast_before)):
                    skip = True
            if f["praise"] and not problem and not f["sugg"]:
                skip = True
        if skip:
            # A clause right after an issue that only adds context ("tried many times, same problem",
            # "and the fees were deducted") is attached to that issue's evidence instead of being lost.
            nc_p = normalize_with_map(text[p.start:p.end])
            adds_context = (find_cues(text[p.start:p.end], lx.CONTINUATION + lx.RECURRING, nc_p)
                            or "PAYMENT" in f["topics"] or "PRIVACY" in f["topics"])
            if issues and last_issue_piece == idx - 1 and adds_context and not f["praise"]:
                ev = text[issue_start[-1]:p.end].strip(" ،,.!؟?\n-")
                issues[-1] = issues[-1].model_copy(update={"evidence_quote": ev})
                last_issue_piece = idx
            continue
        itype = _issue_type(f, topic, p.question, rhetorical_context=neg_context or p.contrast_inside or bool(f["negative"]))
        tdef = lx.TOPICS.get(topic, lx.GENERIC_TOPIC)
        dim = tdef.suggestion_dimension if itype == "Suggestion" else ("INFORMATION_QUALITY" if itype == "Inquiry" else tdef.problem_dimension)
        if topic == "GENERAL_MALFUNCTION" and (itype == "Suggestion" or (f["missing"] and not f["hard"])):
            dim = "FULFILLMENT"
        evidence = text[p.start:p.end].strip(" ،,.!؟?\n-")
        affected = _affected_service(evidence, topic, tdef)
        conf = 0.8 if (f["topics"] and (problem or f["sugg"])) else (0.6 if f["topics"] else 0.5)
        issue_start.append(p.start + (len(text[p.start:p.end]) - len(text[p.start:p.end].lstrip(" ،,.!؟?\n-"))))
        last_issue_piece = idx
        issues.append(LLMIssue(
            description=_description(itype, topic, f), issue_type=itype, dimension=dim,
            evidence_quote=evidence, affected_service=affected,
            missing_information=_missing_info(itype, text, affected), confidence=conf,
        ))

    # Feedback type (first pass is tone-first: dominant tone of the review)
    nc = normalize_with_map(text)
    negative_total = len(find_cues(text, lx.NEGATIVE, nc))
    issue_types = [i.issue_type for i in issues]
    praise_dominant = praise_total > 0 and negative_total == 0 and (review.rating is None or review.rating >= 4)
    if praise_dominant:
        ftype = "Praise"
        secondary = sorted(set(issue_types))
    elif issues:
        res = tools.feedback_type_resolver(issue_types, praise_total > 0)
        ftype, secondary = res["feedback_type"], res["secondary_types"]
    elif negative_total:
        ftype, secondary = "General Complaint", []
    elif find_cues(text, lx.INQUIRY):
        ftype, secondary = "Inquiry", []
    else:
        ftype, secondary = ("Praise" if (review.rating or 0) >= 4 else "General Complaint"), []

    amb = tools.ambiguity_checker(text, [{"topic": "GENERAL_MALFUNCTION" if i.affected_service == "unspecified" else "x", "affected_service": i.affected_service, "issue_type": i.issue_type} for i in issues], ftype)
    reasoning = (
        f"Rule-based reasoner: {len(pieces)} clause segment(s); {len(issues)} issue(s) extracted; "
        f"praise cues={praise_total}, negative cues={negative_total}, rating={review.rating}. "
        + ("Primary type chosen from the dominant (positive) tone." if praise_dominant else "Primary type chosen from extracted issues.")
    )
    return LLMAgent1Analysis(
        feedback_type=ftype, secondary_types=[s for s in secondary if s != ftype], issues=issues,
        is_ambiguous=amb["is_ambiguous"], ambiguity_reasons=amb["reasons"],
        missing_information=amb["missing_information"], reasoning=reasoning,
    )


def infer_topic(evidence: str, issue_type: str = "") -> str:
    t = _primary_topic(topics_in(evidence))
    return t or "GENERAL_MALFUNCTION"


# --------------------------------------------------------------------------- #
# Agent 2 offline reasoning
# --------------------------------------------------------------------------- #
def offline_impact(review: ReviewInput, issues: list) -> LLMImpactAssessment:
    out = []
    review_cues = tools.impact_cue_detector(review.text)
    for iss in issues:
        cues = tools.impact_cue_detector(iss.evidence)
        topic = iss.topic or infer_topic(iss.evidence)
        tdef = lx.TOPICS.get(topic)
        workaround = bool(cues["workaround"] or review_cues["workaround"])
        if iss.issue_type in ("Suggestion", "Inquiry", "Praise") and not cues["hard_failure"]:
            sc, why = "NOT_AFFECTED", f"{iss.issue_type}: no failure reported"
        elif cues["hard_failure"]:
            if cues["partial"] or workaround:
                sc, why = "PARTIALLY_BLOCKED", f"failure cue {cues['hard_failure'][:2]} qualified by {(cues['partial'] + cues['workaround'] + review_cues['workaround'])[:2]}"
            else:
                sc, why = "BLOCKED", f"hard failure cue {cues['hard_failure'][:2]}"
        elif cues["crash"]:
            sc, why = "PARTIALLY_BLOCKED", f"crash cue {cues['crash'][:2]}"
        elif iss.issue_type == "Bug" and (find_cues(iss.evidence, lx.ERROR_CUES) or cues["missing"]) and tools.service_identifier(iss.evidence)["found"]:
            repeated = cues["recurring"] or find_cues(iss.evidence, lx.CONTINUATION)
            sc = "BLOCKED" if repeated else "PARTIALLY_BLOCKED"
            why = f"error/unfulfilled result while performing a service{' - repeated attempts failed' if repeated else ''}"
        elif tdef and tdef.gateway and cues["soft_problem"]:
            sc, why = "PARTIALLY_BLOCKED", f"problem at a gateway step ({tdef.label_en}) {cues['soft_problem'][:2]}"
        elif cues["freeze_or_slow"] or cues["soft_problem"] or cues["missing"] or iss.issue_type == "General Complaint":
            sc, why = "DEGRADED", f"friction cues {(cues['freeze_or_slow'] + cues['soft_problem'] + cues['missing'])[:3] or 'complaint without failure'}"
        else:
            sc, why = "NOT_AFFECTED", "no impact cue"
        scope = "WIDESPREAD" if cues["widespread"] else ("RECURRING" if cues["recurring"] else "INDIVIDUAL")
        risks = list(cues["risk_flags"])
        if tdef and tdef.risk_flag and tdef.risk_flag not in risks and sc != "NOT_AFFECTED":
            risks.append(tdef.risk_flag)
        quote_candidates = cues["hard_failure"] + cues["crash"] + cues["freeze_or_slow"] + cues["soft_problem"] + cues["missing"]
        quote = quote_candidates[0] if quote_candidates else iss.evidence
        summary = {
            "BLOCKED": "Beneficiary cannot complete the service step",
            "PARTIALLY_BLOCKED": "Service step fails intermittently or needs a workaround",
            "DEGRADED": "Service can be completed but with friction",
            "NOT_AFFECTED": "No effect on completing the service",
        }[sc]
        out.append(LLMImpact(
            issue_id=iss.issue_id, service_completion=sc, scope=scope, workaround_mentioned=workaround,
            risk_flags=risks, impact_summary=summary, supporting_quote=quote,
            reasoning=f"{why}; scope={scope} (cues {(cues['widespread'] + cues['recurring'])[:2]}); risks={risks or 'none'}",
        ))
    return LLMImpactAssessment(assessments=out)
