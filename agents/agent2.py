"""Agent 2 - Impact Assessment & Prioritization Agent.

The LLM (or offline reasoner) judges *facts* about impact (does the issue block
the service? how widespread? any financial/data risk?). Severity and priority
are then decided by the documented, auditable rule engine - never by
sentiment or star rating. Agent 2 verifies that the evidence supports its
decision and asks Agent 1 to reassess when it does not.
"""
from __future__ import annotations

import json

import heuristics
from agents.base import BaseAgent
from schemas import (
    SEVERITY_ORDER, Agent2Output, AgentMessage, IssueAssessment, LLMImpact, LLMImpactAssessment,
    Priority, Severity, VerificationStatus, WorkflowState,
)
from tools import severity_rules_kb

_PRIO = ["P1", "P2", "P3", "P4"]


class ImpactAssessmentAgent(BaseAgent):
    name = "Agent2"
    role = "Impact Assessment & Prioritization Agent"
    instructions = """You are Agent 2, the Impact Assessment Agent in a multi-agent framework for Arabic feedback on
Saudi government mobile apps. For EACH issue found by Agent 1, judge its operational impact using the original review:
- service_completion: BLOCKED (cannot complete the service step), PARTIALLY_BLOCKED (fails sometimes or needs a
  workaround), DEGRADED (completed with friction), NOT_AFFECTED (suggestion, inquiry, cosmetic).
- scope: INDIVIDUAL, RECURRING (happens repeatedly), WIDESPREAD (text says many users are affected).
- workaround_mentioned, and risk_flags among FINANCIAL, PERSONAL_DATA_SECURITY, SAFETY, LEGAL_DEADLINE.
- supporting_quote: copy EXACTLY the part of the review that supports your judgement.
Judge impact from what the text states, NOT from how angry the reviewer is or the star rating.
A one-star review can be a Low-impact issue. Do not assume facts not in the text.
Return only the requested structured output."""

    def __init__(self, llm=None):
        super().__init__(llm)
        self._cache: dict[tuple, LLMImpact] = {}

    # ------------------------------------------------------------------ #
    def assess(self, state: WorkflowState) -> list[AgentMessage]:
        """Assess impact; return review requests for Agent 1 (empty list = done)."""
        r, a1 = state.review, state.agent1
        state.log(self.name, "START", f"Assessing {len(a1.issues)} issue(s) from Agent1 (round {state.revision_count})")
        senti = self.call_tool(state, "sentiment_intensity", "Measure sentiment separately from severity",
                               review_text=r.text, rating=r.rating)

        if not a1.issues:
            note = (f"Sentiment is '{senti['label']}' (rating {r.rating}) but no operational issue was identified, "
                    "so no severity is assigned (rule S0: sentiment never sets severity).")
            state.log(self.name, "DECISION", note)
            state.agent2 = Agent2Output(sentiment_intensity=senti["label"], sentiment_score=senti["score"],
                                        sentiment_vs_severity_note=note, reasoning_summary=note,
                                        verification_status=VerificationStatus.NOT_APPLICABLE, reasoner="rules")
            state.messages.append(AgentMessage(sender=self.name, receiver="Agent3", message_type="HANDOFF", content=note))
            return []

        impacts, reasoner = self._impacts(state)
        assessments: list[IssueAssessment] = []
        concerns: list[tuple[str, list[str], str]] = []
        for iss in a1.issues:
            imp = impacts[iss.issue_id]
            ev = [iss.evidence]
            qv = self.call_tool(state, "evidence_verifier", f"Verify impact quote for {iss.issue_id}",
                                review_text=r.text, evidence=imp.supporting_quote)
            if qv["verified"] and imp.supporting_quote not in iss.evidence:
                ev.append(imp.supporting_quote)
            elif not qv["verified"]:
                state.log(self.name, "CHECK", f"{iss.issue_id}: impact quote not found in review; relying on Agent1 evidence only")

            # Grounding check for blocking claims (conditional tool use)
            if imp.service_completion in ("BLOCKED", "PARTIALLY_BLOCKED"):
                cues = self.call_tool(state, "impact_cue_detector", f"{iss.issue_id} judged {imp.service_completion}: confirm a failure cue exists in the evidence",
                                      span=iss.evidence + " " + (imp.supporting_quote if qv["verified"] else ""))
                if not (cues["hard_failure"] or cues["crash"] or cues["soft_problem"]):
                    concerns.append(("WEAK_EVIDENCE", [iss.issue_id],
                                     f"Impact judged {imp.service_completion} but the quoted evidence contains no explicit failure statement. "
                                     "Please re-examine the original text for the span that states the failure."))
            if not iss.evidence_verified:
                concerns.append(("WEAK_EVIDENCE", [iss.issue_id], "Agent1 evidence could not be verified against the review."))

            sev = self.call_tool(state, "severity_rule_engine", f"Apply proposed severity rules to {iss.issue_id}",
                                 service_completion=imp.service_completion, scope=imp.scope, risk_flags=list(imp.risk_flags))
            pr = self.call_tool(state, "priority_rule_engine", f"Apply proposed priority rules to {iss.issue_id}",
                                severity=sev["severity"], scope=imp.scope)
            assessments.append(IssueAssessment(
                issue_id=iss.issue_id, issue_description=iss.description, service_completion=imp.service_completion,
                scope=imp.scope, workaround_mentioned=imp.workaround_mentioned, risk_flags=list(imp.risk_flags),
                impact_summary=imp.impact_summary, severity=Severity(sev["severity"]), priority=Priority(pr["priority"]),
                rules_applied=[sev["rule"]] + pr["rules"], reasoning=f"{imp.reasoning} | Rule {sev['rule']}: {sev['explanation']}",
                evidence=ev, verification_status=VerificationStatus.VERIFIED,
            ))

        # Cross-checks against Agent 1's classification
        blocking = [a for a in assessments if a.service_completion in ("BLOCKED", "PARTIALLY_BLOCKED")]
        if a1.feedback_type in ("Praise", "Suggestion", "Inquiry") and blocking:
            concerns.append(("TYPE_CONFLICT", [a.issue_id for a in blocking],
                             f"Primary type is '{a1.feedback_type}' but issue(s) {[a.issue_id for a in blocking]} block or partially block "
                             "the service. Please confirm or revise the feedback type."))
        for a in assessments:
            iss = next(i for i in a1.issues if i.issue_id == a.issue_id)
            if a1.is_ambiguous and SEVERITY_ORDER[a.severity.value] >= SEVERITY_ORDER["High"]:
                concerns.append(("AMBIGUITY_CHECK", [a.issue_id],
                                 f"{a.issue_id} would be {a.severity.value} but Agent1 flagged the review as ambiguous. "
                                 "Is the failure specific enough to support this severity?"))
            if SEVERITY_ORDER[a.severity.value] >= SEVERITY_ORDER["High"] and iss.affected_service == "unspecified":
                concerns.append(("MISSING_SERVICE_CONTEXT", [a.issue_id],
                                 f"{a.issue_id} is {a.severity.value} but the affected service is unspecified. Please re-check the review."))
        if len(a1.issues) > 1:
            dups = self.call_tool(state, "issue_deduplicator", "Check Agent1's issues for duplicates before prioritising",
                                  issues=[i.model_dump() for i in a1.issues])["duplicates"]
            for x, y in dups:
                concerns.append(("DUPLICATE_ISSUES", [x, y], f"{x} and {y} appear to describe the same problem."))

        # Decide: request reassessment, or finalise
        # A concern Agent1 already CONFIRMED is not re-sent; one it REVISED but that persists may be re-sent.
        pending = self._dedupe([c for c in concerns if state.resolved_concerns.get(self._key(c)) != "CONFIRMED"])
        if pending and state.revision_count < state.max_revisions:
            msgs = [AgentMessage(sender=self.name, receiver="Agent1", message_type="REVIEW_REQUEST", round=state.revision_count + 1,
                                 concern_code=code, issue_ids=ids, content=text) for code, ids, text in pending]
            state.log(self.name, "REQUEST_REASSESSMENT",
                      f"Round {state.revision_count + 1}: sending {len(msgs)} concern(s) to Agent1: " + ", ".join(m.concern_code for m in msgs))
            state.agent2 = self._output(state, assessments, senti, reasoner, provisional=True)
            return msgs

        if pending:
            state.log(self.name, "LIMIT_REACHED", f"Max revisions ({state.max_revisions}) reached with {len(pending)} unresolved concern(s); "
                      "finalising with INSUFFICIENT_EVIDENCE where affected")
        self._finalise_verification(state, assessments, concerns, pending)
        out = self._output(state, assessments, senti, reasoner, provisional=False)
        state.agent2 = out
        state.log(self.name, "DECISION", out.reasoning_summary)
        state.messages.append(AgentMessage(sender=self.name, receiver="Agent3", message_type="HANDOFF",
                                           content=f"Overall severity {out.overall_severity.value if out.overall_severity else '-'}, "
                                                   f"priority {out.overall_priority.value if out.overall_priority else '-'}, verification {out.verification_status.value}",
                                           issue_ids=[a.issue_id for a in assessments]))
        return []

    # ------------------------------------------------------------------ #
    def _impacts(self, state: WorkflowState) -> tuple[dict[str, LLMImpact], str]:
        r, issues = state.review, state.agent1.issues
        key = lambda i: (r.review_id, i.issue_id, i.evidence, i.issue_type)  # noqa: E731
        todo = [i for i in issues if key(i) not in self._cache]
        reasoner = "cached"
        if todo:
            payload = [{"issue_id": i.issue_id, "description": i.description, "issue_type": i.issue_type,
                        "dimension": i.dimension, "evidence": i.evidence, "affected_service": i.affected_service} for i in todo]
            user = (f"Original review (app={r.app_id}, rating={r.rating}):\n<<<\n{r.text}\n>>>\n\n"
                    f"Agent 1 issues to assess:\n{json.dumps(payload, ensure_ascii=False, indent=1)}")
            res, reasoner = self.reason(state, f"impact assessment of {[i.issue_id for i in todo]}", user, LLMImpactAssessment,
                                        lambda: heuristics.offline_impact(r, todo))
            by_id = {a.issue_id: a for a in res.assessments}
            fallback = None
            for i in todo:
                imp = by_id.get(i.issue_id)
                if imp is None:  # LLM skipped an issue -> rule-based fallback for that issue
                    fallback = fallback or {a.issue_id: a for a in heuristics.offline_impact(r, todo).assessments}
                    imp = fallback[i.issue_id]
                    state.log(self.name, "CHECK", f"{i.issue_id} missing from reasoning output; rule-based impact used")
                self._cache[key(i)] = imp
        else:
            state.log(self.name, "CACHE", "Issues unchanged since last round; reusing impact judgements")
        return {i.issue_id: self._cache[key(i)] for i in issues}, reasoner

    @staticmethod
    def _key(c) -> str:
        return f"{c[0]}:{c[1][0] if c[1] else '*'}"

    @staticmethod
    def _dedupe(concerns):
        seen, out = set(), []
        for c in concerns:
            k = (c[0], tuple(c[1]))
            if k not in seen:
                seen.add(k)
                out.append(c)
        return out

    def _finalise_verification(self, state, assessments, concerns, unresolved) -> None:
        raised_ids = {k.split(":", 1)[1] for k in state.resolved_concerns}
        weak_codes = {"WEAK_EVIDENCE", "AMBIGUITY_CHECK"}
        insufficient = set()
        for c in concerns:
            res = state.resolved_concerns.get(self._key(c))
            if c in unresolved or (c[0] in weak_codes and res == "CONFIRMED"):
                insufficient.update(c[1])
        cap = severity_rules_kb().get("insufficient_evidence_priority_cap", "P3")
        for a in assessments:
            if a.issue_id in insufficient:
                a.verification_status = VerificationStatus.INSUFFICIENT_EVIDENCE
                a.provisional = True
                if _PRIO.index(a.priority.value) < _PRIO.index(cap):
                    a.priority = Priority(cap)
                    a.rules_applied.append("P-VERIFY")
                a.reasoning += " | Evidence remained insufficient after the revision loop: severity is provisional, priority capped."
            elif a.issue_id in raised_ids or "*" in raised_ids:
                a.verification_status = VerificationStatus.VERIFIED_AFTER_REVISION

    def _output(self, state, assessments, senti, reasoner, provisional: bool) -> Agent2Output:
        overall_sev = max((a.severity for a in assessments), key=lambda s: SEVERITY_ORDER[s.value]) if assessments else None
        overall_pr = min((a.priority for a in assessments), key=lambda p: _PRIO.index(p.value)) if assessments else None
        statuses = {a.verification_status for a in assessments}
        if VerificationStatus.INSUFFICIENT_EVIDENCE in statuses:
            vs = VerificationStatus.INSUFFICIENT_EVIDENCE
        elif VerificationStatus.VERIFIED_AFTER_REVISION in statuses:
            vs = VerificationStatus.VERIFIED_AFTER_REVISION
        else:
            vs = VerificationStatus.VERIFIED
        note = ""
        if overall_sev is not None:
            neg = senti["score"] <= -0.2
            if neg and SEVERITY_ORDER[overall_sev.value] <= 2:
                note = (f"Sentiment is '{senti['label']}' (rating {state.review.rating}) but operational severity is only "
                        f"{overall_sev.value}: strong negative tone does not by itself make an issue severe (rule S0).")
            elif not neg and SEVERITY_ORDER[overall_sev.value] >= 3:
                note = (f"Sentiment is '{senti['label']}' yet severity is {overall_sev.value}: the issue blocks the service "
                        "even though the tone is mild.")
            else:
                note = f"Sentiment '{senti['label']}' and severity {overall_sev.value} are consistent; severity derived from impact rules only."
        summary = "; ".join(f"{a.issue_id}: {a.service_completion}/{a.scope} -> {a.severity.value} {a.priority.value} ({','.join(a.rules_applied)})"
                            for a in assessments)
        return Agent2Output(
            assessments=assessments, overall_severity=overall_sev, overall_priority=overall_pr,
            sentiment_intensity=senti["label"], sentiment_score=senti["score"], sentiment_vs_severity_note=note,
            verification_status=vs, review_requests_sent=sum(1 for m in state.messages if m.message_type == "REVIEW_REQUEST"),
            reasoning_summary=("[PROVISIONAL] " if provisional else "") + summary, reasoner=reasoner,
        )
