"""Agent 1 - Intelligent Feedback Investigation Agent.

Responsibilities: understand the Arabic review (incl. Saudi dialect), classify
the feedback type, extract *independent* issues with exact evidence, map each
issue to an EGSQUAL dimension through the knowledge-base tool, detect
ambiguity, verify its own output, decide ACCEPT / REVISE / NEEDS_CLARIFICATION,
and answer review requests from Agent 2.
"""
from __future__ import annotations

import json

import heuristics
from agents.base import BaseAgent
from schemas import (
    Agent1Output, AgentMessage, AnalysisStatus, Issue, LLMAgent1Analysis, LLMIssue,
    LLMRevisionResponse, WorkflowState,
)
from arabic_text import word_count


class FeedbackInvestigationAgent(BaseAgent):
    name = "Agent1"
    role = "Intelligent Feedback Investigation Agent"
    instructions = """You are Agent 1, the Feedback Investigation Agent in a multi-agent framework that analyses
Arabic beneficiary reviews of Saudi government mobile apps (MSA and Saudi dialect).

Your job:
1. Read the ORIGINAL Arabic review carefully, including Saudi dialect (e.g. يعلق = freezes, ما يشتغل = does not work).
2. Choose the primary feedback type: Bug, Suggestion, Inquiry, Praise, or General Complaint (and secondary types).
3. Extract every INDEPENDENT issue separately (one problem per issue). Do not merge different problems,
   and do not split one problem into several.
4. For each issue: short English description, issue_type, EGSQUAL dimension (use ONLY the provided keys),
   an evidence_quote copied EXACTLY (verbatim substring) from the review, the affected service/feature
   (or 'unspecified'), and any missing information needed to act.
5. Flag the review as ambiguous when it does not say what failed, is too short, or is global/unspecific.
6. Never invent facts that are not in the text. Praise alone is not an issue.
Return only the requested structured output."""

    # ------------------------------------------------------------------ #
    def analyze(self, state: WorkflowState) -> Agent1Output:
        r = state.review
        state.log(self.name, "START", f"Investigating review {r.review_id} (app={r.app_id}, rating={r.rating})")
        self.call_tool(state, "arabic_normalizer", "Normalise Arabic orthography before analysis", text=r.text)
        dialect = self.call_tool(state, "dialect_lexicon_lookup", "Detect Saudi-dialect expressions", text=r.text)["dialect_terms"]

        prompt_parts = [f"Review (app={r.app_id}, rating={r.rating}):\n<<<\n{r.text}\n>>>"]
        if self.llm.available:
            kb = self.call_tool(state, "egsqual_kb_lookup", "Load EGSQUAL dimension definitions for the reasoning context",
                                dimension_key="ALL")["dimensions"]
            prompt_parts.append("EGSQUAL dimensions (use these keys):\n" + json.dumps(kb, ensure_ascii=False, indent=1))
        if dialect:
            state.log(self.name, "DECISION", f"Dialect detected ({', '.join(dialect)}); adding glossary to the reasoning context")
            prompt_parts.append("Dialect glossary for terms found in this review:\n" + json.dumps(dialect, ensure_ascii=False))

        raw, reasoner = self.reason(state, "issue extraction", "\n\n".join(prompt_parts), LLMAgent1Analysis,
                                    lambda: heuristics.offline_agent1_analysis(r))
        out = self._verify_and_decide(state, raw, reasoner, dialect, revision_round=0)
        state.agent1 = out
        state.agent1_history.append(out.model_copy(deep=True))
        state.messages.append(AgentMessage(
            sender=self.name, receiver="Agent2", message_type="HANDOFF",
            content=f"Analysis {out.status.value}: type={out.feedback_type}, {len(out.issues)} issue(s)",
            issue_ids=[i.issue_id for i in out.issues]))
        return out

    # ------------------------------------------------------------------ #
    def _verify_and_decide(self, state: WorkflowState, raw: LLMAgent1Analysis, reasoner: str,
                           dialect: dict, revision_round: int, previous: Agent1Output | None = None) -> Agent1Output:
        r = state.review
        self_revisions: list[str] = []
        issues: list[Issue] = []
        prev_ids = {i.description: i.issue_id for i in (previous.issues if previous else [])}
        next_id = 1 + max([int(i.issue_id[1:]) for i in (previous.issues if previous else [])] or [0])

        for k, li in enumerate(raw.issues):
            issue_id = prev_ids.get(li.description) or (f"I{k + 1}" if previous is None else f"I{next_id}")
            if previous is not None and li.description not in prev_ids:
                next_id += 1
            issue = self._build_issue(state, li, issue_id, self_revisions)
            if issue is not None:
                issues.append(issue)

        # duplicate detection (conditional: only meaningful with 2+ issues)
        if len(issues) > 1:
            dups = self.call_tool(state, "issue_deduplicator", "Check whether two extracted issues describe the same problem",
                                  issues=[i.model_dump() for i in issues])["duplicates"]
            for a, b in dups:
                keep = next((i for i in issues if i.issue_id == a), None)
                drop = next((i for i in issues if i.issue_id == b), None)
                if keep and drop:
                    if len(drop.evidence) > len(keep.evidence):
                        keep.evidence = drop.evidence
                    issues.remove(drop)
                    self_revisions.append(f"merged duplicate issue {b} into {a}")
                    state.log(self.name, "SELF_REVISION", f"Merged duplicate issue {b} into {a}")

        # ambiguity check (conditional)
        is_amb, reasons, missing = raw.is_ambiguous, list(raw.ambiguity_reasons), list(raw.missing_information)
        if len(issues) <= 1 or word_count(r.text) <= 6 or raw.is_ambiguous:
            amb = self.call_tool(state, "ambiguity_checker", "Few issues / short text: check for ambiguous or incomplete feedback",
                                 review_text=r.text,
                                 issues=[{"topic": i.topic, "affected_service": i.affected_service} for i in issues],
                                 feedback_type=raw.feedback_type)
            if amb["is_ambiguous"] != is_amb:
                state.log(self.name, "SELF_REVISION", f"Ambiguity flag set to {amb['is_ambiguous']} by ambiguity checker")
            is_amb = amb["is_ambiguous"] or (raw.is_ambiguous and not issues)
            reasons = sorted(set(reasons + amb["reasons"]))
            missing = sorted(set(missing + amb["missing_information"]))

        if is_amb:
            status = AnalysisStatus.NEEDS_CLARIFICATION
            why = "Feedback is ambiguous/incomplete: " + "; ".join(reasons)
        elif self_revisions:
            status = AnalysisStatus.REVISED
            why = "Analysis revised after self-verification: " + "; ".join(self_revisions)
        else:
            status = AnalysisStatus.ACCEPTED
            why = "All issues are supported by verified evidence and mapped to valid EGSQUAL dimensions"
        state.log(self.name, "DECISION", f"Status {status.value}. {why}")

        return Agent1Output(
            feedback_type=raw.feedback_type, secondary_types=list(raw.secondary_types), issues=issues,
            is_ambiguous=is_amb, ambiguity_reasons=reasons, missing_information=missing, status=status,
            reasoning=raw.reasoning, dialect_terms=dialect, revision_round=revision_round, reasoner=reasoner,
        )

    def _build_issue(self, state: WorkflowState, li: LLMIssue, issue_id: str, self_revisions: list[str]) -> Issue | None:
        r = state.review
        kb = self.call_tool(state, "egsqual_kb_lookup", f"Look up definition of dimension {li.dimension} for {issue_id}",
                            dimension_key=li.dimension)
        dim = li.dimension
        if not kb["found"]:
            sug = self.call_tool(state, "egsqual_kb_search", f"Dimension '{li.dimension}' not in KB - search by evidence keywords",
                                 text=li.evidence_quote)
            dim = sug["best_dimension"]
            kb = self.call_tool(state, "egsqual_kb_lookup", f"Look up suggested dimension {dim}", dimension_key=dim)
            self_revisions.append(f"{issue_id}: dimension corrected to {dim}")

        evidence = li.evidence_quote
        ver = self.call_tool(state, "evidence_verifier", f"Verify evidence for {issue_id} is in the original review",
                             review_text=r.text, evidence=evidence)
        topic = heuristics.infer_topic(evidence)
        if not ver["verified"]:
            loc = self.call_tool(state, "evidence_locator", f"Evidence for {issue_id} not verified - search the review for a supporting span",
                                 review_text=r.text, topic_key=topic, current_evidence=evidence)
            if loc["found"]:
                evidence = loc["evidence"]
                ver = self.call_tool(state, "evidence_verifier", f"Re-verify relocated evidence for {issue_id}",
                                     review_text=r.text, evidence=evidence)
                self_revisions.append(f"{issue_id}: evidence replaced with verified span")
                state.log(self.name, "SELF_REVISION", f"{issue_id}: unsupported quote replaced by verified span")
            if not ver["verified"]:
                state.log(self.name, "REJECT_ISSUE", f"{issue_id} ('{li.description}') rejected: no supporting evidence in the review")
                self_revisions.append(f"{issue_id}: rejected as unsupported")
                return None
        return Issue(
            issue_id=issue_id, description=li.description, issue_type=li.issue_type, topic=topic, dimension=dim,
            dimension_name=kb.get("name", dim), dimension_definition=kb.get("definition", ""), evidence=evidence,
            evidence_verified=ver["verified"], evidence_score=ver["score"], affected_service=li.affected_service or "unspecified",
            missing_information=list(li.missing_information), confidence=li.confidence,
        )

    # ------------------------------------------------------------------ #
    # Inter-agent: respond to Agent 2's review requests
    # ------------------------------------------------------------------ #
    def respond(self, state: WorkflowState, requests: list[AgentMessage]) -> list[AgentMessage]:
        rnd = state.revision_count + 1
        prev = state.agent1
        state.log(self.name, "REVIEW_RECEIVED", f"Round {rnd}: {len(requests)} concern(s) from Agent2: "
                  + "; ".join(f"{m.concern_code}{m.issue_ids}" for m in requests))
        if self.llm.available:
            responses = self._respond_llm(state, requests, prev, rnd)
            if responses is not None:
                return responses
        return self._respond_tools(state, requests, prev, rnd)

    def _respond_llm(self, state, requests, prev: Agent1Output, rnd: int) -> list[AgentMessage] | None:
        r = state.review
        user = (f"Original review (app={r.app_id}, rating={r.rating}):\n<<<\n{r.text}\n>>>\n\n"
                f"Your previous analysis:\n{prev.model_dump_json(include={'feedback_type', 'secondary_types', 'issues', 'is_ambiguous', 'missing_information'})}\n\n"
                "Agent 2 (Impact Assessment) raised these concerns:\n"
                + "\n".join(f"- [{m.concern_code}] issues {m.issue_ids}: {m.content}" for m in requests)
                + "\n\nRe-examine the ORIGINAL review evidence. Decide CONFIRM (keep your analysis) or REVISE, "
                  "explain why, and return the full (possibly revised) analysis. Keep issue descriptions unchanged for issues you keep.")
        try:
            resp = self.llm.structured(self.instructions, user, LLMRevisionResponse)
        except Exception as e:  # LLMError or validation issue
            state.errors.append(f"Agent1/revision: {e}")
            state.log(self.name, "LLM_FALLBACK", f"Revision via LLM failed ({str(e)[:100]}); using tool-based revision")
            return None
        state.log(self.name, "LLM_REASONING", f"Revision round {rnd}: {resp.decision} - {resp.explanation[:160]}")
        new = self._verify_and_decide(state, resp.revised_analysis, self.llm.name, prev.dialect_terms, rnd, previous=prev)
        if resp.decision == "REVISE" and new.status == AnalysisStatus.ACCEPTED:
            new.status = AnalysisStatus.REVISED
        self._commit(state, new)
        out = []
        for m in requests:
            for iid in (m.issue_ids or ["*"]):
                state.resolved_concerns[f"{m.concern_code}:{iid}"] = "REVISED" if resp.decision == "REVISE" else "CONFIRMED"
            out.append(self._reply(m, rnd, resp.decision, resp.explanation))
        return out

    def _respond_tools(self, state, requests, prev: Agent1Output, rnd: int) -> list[AgentMessage]:
        r = state.review
        new = prev.model_copy(deep=True)
        new.revision_round = rnd
        replies = []
        for m in requests:
            code = m.concern_code
            decision, expl = "CONFIRM", ""
            targets = [i for i in new.issues if i.issue_id in m.issue_ids]
            if code == "TYPE_CONFLICT":
                res = self.call_tool(state, "feedback_type_resolver", "Re-derive feedback type from the extracted issues",
                                     issue_types=[i.issue_type for i in new.issues], has_praise=prev.feedback_type == "Praise" or "Praise" in prev.secondary_types)
                if res["feedback_type"] != new.feedback_type:
                    expl = (f"Primary type changed {new.feedback_type} -> {res['feedback_type']}: the review contains an operational "
                            f"problem; the positive tone is kept as a secondary type.")
                    new.feedback_type, new.secondary_types = res["feedback_type"], res["secondary_types"]
                    decision = "REVISE"
                else:
                    expl = f"Type {new.feedback_type} is consistent with the extracted issues."
            elif code == "WEAK_EVIDENCE":
                for iss in targets:
                    loc = self.call_tool(state, "evidence_locator", f"Re-examine original text for stronger evidence for {iss.issue_id}",
                                         review_text=r.text, topic_key=iss.topic, current_evidence=iss.evidence)
                    if loc["found"] and loc["evidence"] != iss.evidence and loc.get("contains_problem_marker"):
                        ver = self.call_tool(state, "evidence_verifier", f"Verify new evidence for {iss.issue_id}", review_text=r.text, evidence=loc["evidence"])
                        if ver["verified"]:
                            iss.evidence, iss.evidence_verified, iss.evidence_score = loc["evidence"], True, ver["score"]
                            iss.revision_note = f"Evidence extended in round {rnd}"
                            decision, expl = "REVISE", f"{iss.issue_id}: evidence extended to the clause that states the failure."
                            continue
                    expl = f"{iss.issue_id}: the review contains no stronger evidence; original quote kept."
            elif code == "AMBIGUITY_CHECK":
                amb = self.call_tool(state, "ambiguity_checker", "Re-check whether the review states what failed",
                                     review_text=r.text, issues=[{"topic": i.topic, "affected_service": i.affected_service} for i in new.issues],
                                     feedback_type=new.feedback_type)
                if amb["is_ambiguous"]:
                    new.missing_information = sorted(set(new.missing_information + amb["missing_information"]))
                    expl = "Confirmed: the text does not specify the failing feature/service; clarification is required."
                else:
                    new.is_ambiguous, decision = False, "REVISE"
                    expl = "On re-examination the review is specific enough; ambiguity flag removed."
            elif code == "MISSING_SERVICE_CONTEXT":
                svc = self.call_tool(state, "service_identifier", "Search the whole review for the affected service", text=r.text)
                for iss in targets:
                    if svc["found"]:
                        iss.affected_service = svc["affected_service"]
                        iss.revision_note = f"Affected service identified in round {rnd}"
                        decision, expl = "REVISE", f"{iss.issue_id}: affected service identified as '{svc['affected_service']}'."
                    else:
                        if "Which service / transaction was being performed" not in iss.missing_information:
                            iss.missing_information.append("Which service / transaction was being performed")
                        expl = f"{iss.issue_id}: the review does not name the service; recorded as missing information."
            elif code == "DUPLICATE_ISSUES" and len(targets) == 2:
                a, b = targets
                a.evidence = a.evidence if len(a.evidence) >= len(b.evidence) else b.evidence
                new.issues = [i for i in new.issues if i.issue_id != b.issue_id]
                decision, expl = "REVISE", f"Merged {b.issue_id} into {a.issue_id} (same problem)."
            for iid in (m.issue_ids or ["*"]):
                state.resolved_concerns[f"{code}:{iid}"] = "REVISED" if decision == "REVISE" else "CONFIRMED"
            state.log(self.name, "REVISION_DECISION", f"[{code}] {decision}: {expl}")
            replies.append(self._reply(m, rnd, decision, expl))

        if any(x.content.startswith("REVISE") for x in replies):
            new.status = AnalysisStatus.NEEDS_CLARIFICATION if new.is_ambiguous else AnalysisStatus.REVISED
        self._commit(state, new)
        return replies

    def _commit(self, state: WorkflowState, new: Agent1Output) -> None:
        state.agent1 = new
        state.agent1_history.append(new.model_copy(deep=True))
        state.log(self.name, "DECISION", f"Round {new.revision_round} result: status={new.status.value}, type={new.feedback_type}, issues={len(new.issues)}")

    def _reply(self, req: AgentMessage, rnd: int, decision: str, expl: str) -> AgentMessage:
        msg = AgentMessage(sender=self.name, receiver="Agent2", message_type="REVIEW_RESPONSE", round=rnd,
                           concern_code=req.concern_code, issue_ids=req.issue_ids, content=f"{decision}: {expl}")
        return msg
