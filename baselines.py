"""Comparison baselines for the overall-framework evaluation.

Both baselines get the SAME domain knowledge as the multi-agent framework
(EGSQUAL definitions, the severity/priority rule descriptions, the escalation
rule descriptions and the action definitions) so that the comparison measures
the architecture, not the information available:

* Single LLM - one structured call does everything at once.
* Single LLM (few-shot + CoT) - the strongest single-call baseline: worked examples and step-by-step
  reasoning before the answer.
* Sequential LLM pipeline - three fixed LLM calls (classify -> assess ->
  decide), each feeding the next.

Unlike the multi-agent framework, neither baseline has tools (no evidence
verifier, no rule engines, no complaint-history / SLA lookup), no
self-verification and no Agent 1 <-> Agent 2 review loop.

Baselines need a working LLM. They are never simulated: if the LLM is not
available or a call fails, the review is recorded as FAILED.
"""
from __future__ import annotations

import json
import time
from datetime import date
from typing import Callable, Literal, Optional

from pydantic import BaseModel, Field

import tools
from evaluation_suite import ACTION_CODES, Prediction
from llm import LLMClient, LLMError
from schemas import DimensionLit, FeedbackTypeLit, ReviewInput, SimulatedTicket

ActionLit = Literal["TECH_TICKET", "SERVICE_IMPROVEMENT", "REQUEST_INFO", "FOLLOW_UP", "ESCALATION_CANDIDATE", "NO_ACTION"]
SentimentLit = Literal["Negative", "Neutral", "Positive"]
SeverityLit = Literal["None", "Low", "Medium", "High", "Critical"]
PriorityLit = Literal["None", "P1", "P2", "P3", "P4"]
CODE_TO_ACTION = {v: k for k, v in ACTION_CODES.items()}


class BaselineTicket(BaseModel):
    title: str
    affected_service: str
    issue_description: str
    severity: str = Field(description="Low, Medium, High or Critical")
    priority: str = Field(description="P1, P2, P3 or P4")
    justification: str
    evidence: list[str] = Field(default_factory=list, description="Exact quotes from the review")


class SingleLLMOutput(BaseModel):
    feedback_type: FeedbackTypeLit
    dimensions: list[DimensionLit] = Field(default_factory=list, description="EGSQUAL keys of the issues; empty if no issue")
    sentiment: SentimentLit
    severity: SeverityLit
    priority: PriorityLit
    action: ActionLit
    escalation_candidate: bool
    evidence: list[str] = Field(default_factory=list, description="Exact quotes from the review supporting the issues")
    ticket: Optional[BaselineTicket] = Field(default=None, description="Required unless action is NO_ACTION")
    reasoning: str = ""


class SingleLLMCoTOutput(BaseModel):
    """Same fields as SingleLLMOutput, but the step-by-step reasoning is written FIRST (chain of thought)."""
    reasoning: str = Field(description="Think step by step: issues, evidence, impact on completing the service, rules, decision")
    feedback_type: FeedbackTypeLit
    dimensions: list[DimensionLit] = Field(default_factory=list)
    sentiment: SentimentLit
    severity: SeverityLit
    priority: PriorityLit
    action: ActionLit
    escalation_candidate: bool
    evidence: list[str] = Field(default_factory=list, description="Exact quotes from the review")
    ticket: Optional[BaselineTicket] = None


class StepClassify(BaseModel):
    feedback_type: FeedbackTypeLit
    dimensions: list[DimensionLit] = Field(default_factory=list)
    issues: list[str] = Field(default_factory=list, description="Short English description of each independent issue")
    evidence: list[str] = Field(default_factory=list, description="Exact quotes from the review")


class StepAssess(BaseModel):
    sentiment: SentimentLit
    severity: SeverityLit
    priority: PriorityLit
    reasoning: str = ""


class StepDecide(BaseModel):
    action: ActionLit
    escalation_candidate: bool
    ticket: Optional[BaselineTicket] = None
    reasoning: str = ""


# --------------------------------------------------------------------------- #
# Shared knowledge (identical to what the agents can consult)
# --------------------------------------------------------------------------- #
def knowledge() -> str:
    kb = {k: {"name": d["name_en"], "definition": d["definition"]} for k, d in tools.egsqual_kb().items()}
    sr, er = tools.severity_rules_kb(), tools.escalation_rules_kb()
    actions = {
        "TECH_TICKET": "A bug that affects completing the service -> internal technical ticket",
        "SERVICE_IMPROVEMENT": "Suggestion / friction that does not block the service -> service improvement request",
        "REQUEST_INFO": "Unclear or unspecific problem -> ask the beneficiary for more information",
        "FOLLOW_UP": "Beneficiary mentions a previous complaint but escalation rules are not all met -> follow up on it",
        "ESCALATION_CANDIDATE": "All escalation rules E1-E4 met -> SIMULATED escalation candidate, needs human approval",
        "NO_ACTION": "Praise / no operational issue",
    }
    return "\n\n".join([
        "EGSQUAL dimensions:\n" + json.dumps(kb, ensure_ascii=False, indent=1),
        "Severity & priority principle: " + sr["principle"],
        "Impact factors:\n" + json.dumps(sr["impact_factors"], ensure_ascii=False, indent=1),
        "Severity rules:\n" + json.dumps(sr["severity_rules"], ensure_ascii=False, indent=1),
        "Priority rules:\n" + json.dumps(sr["priority_rules"], ensure_ascii=False, indent=1),
        "Escalation rules (all of E1-E4 must hold; minimum severity " + er["min_severity_for_escalation"] + "):\n"
        + json.dumps(er["rules"], ensure_ascii=False, indent=1),
        "Actions:\n" + json.dumps(actions, ensure_ascii=False, indent=1),
    ])


ROLE = ("You analyse Arabic beneficiary reviews (MSA and Saudi dialect) of Saudi government mobile apps. "
        "Use ONLY the given definitions and rules. Do not invent complaint numbers, dates or facts. "
        "Severity is 'None' and priority 'None' when there is no operational issue. "
        "Provide a ticket for every action except NO_ACTION. Escalation is simulated only.")


def _review_block(r: ReviewInput) -> str:
    return f"Review (app={r.app_id}, rating={r.rating}):\n<<<\n{r.text}\n>>>"


def _ticket_json(r: ReviewInput, action: str, esc: bool, t: Optional[BaselineTicket], dims: list[str]) -> Optional[str]:
    """Assemble the baseline's ticket into the same SimulatedTicket JSON format the framework uses."""
    if t is None:
        return None
    data = SimulatedTicket(
        ticket_id=f"BASE-{r.review_id}", ticket_type=action, title=t.title, application=f"{r.app_name} ({r.app_id})",
        affected_service=t.affected_service, issue_description=t.issue_description, egsqual_dimensions=sorted(dims),
        severity=t.severity, priority=t.priority, recommended_action=CODE_TO_ACTION[action].value,
        justification=t.justification, evidence=t.evidence,
        escalation_status="CANDIDATE_AWAITING_HUMAN_APPROVAL" if esc else "NOT_ELIGIBLE",
        human_approval_required=True, original_review_id=r.review_id,
    )
    return data.model_dump_json()


# Worked examples for the few-shot baseline. They are written for this purpose and are NOT taken from the
# dataset, so no test review leaks into the prompt.
FEW_SHOT = """Worked examples (invented, not from the test set):

Review: "كل ما ادخل يطلع لي خطأ في النظام وما قدرت اسدد المخالفة، والمبلغ انخصم مرتين"
-> feedback_type=Bug, dimensions=[SYSTEM_AVAILABILITY, FULFILLMENT], sentiment=Negative,
   severity=Critical (payment blocked AND money deducted twice = financial risk), priority=P1, action=TECH_TICKET,
   escalation_candidate=false (no previous complaint mentioned), evidence=["يطلع لي خطأ في النظام", "المبلغ انخصم مرتين"]

Review: "ياليت تضيفون الوضع الليلي، التطبيق ممتاز"
-> feedback_type=Suggestion, dimensions=[EFFICIENCY], sentiment=Positive, severity=Low, priority=P4,
   action=SERVICE_IMPROVEMENT, escalation_candidate=false, evidence=["ياليت تضيفون الوضع الليلي"]

Review: "سيء"
-> feedback_type=General Complaint, dimensions=[], sentiment=Negative, severity=None, priority=None,
   action=REQUEST_INFO (nothing concrete to act on), escalation_candidate=false, evidence=[]

Review: "رفعت شكوى رقم 12345 قبل شهر وللحين ما احد رد، والتطبيق للحين ما يفتح"
-> feedback_type=Bug, dimensions=[SYSTEM_AVAILABILITY, RESPONSIVENESS], sentiment=Negative, severity=High, priority=P2,
   action=FOLLOW_UP (a complaint is mentioned but its status and SLA cannot be verified, so it is not yet an escalation
   candidate), escalation_candidate=false, evidence=["التطبيق للحين ما يفتح", "رفعت شكوى رقم 12345 قبل شهر"]
"""


def single_llm_cot(llm: LLMClient, r: ReviewInput) -> Prediction:
    p = Prediction(system="single_llm_cot", review_id=r.review_id)
    out = llm.structured(ROLE + "\n\n" + knowledge() + "\n\n" + FEW_SHOT,
                         _review_block(r) + "\n\nReason step by step in `reasoning` first, then give the complete analysis.",
                         SingleLLMCoTOutput)
    _fill(p, r, out, out, out)
    return p


def _fill(p: Prediction, r: ReviewInput, cls, assess, decide) -> None:
    p.feedback_type = cls.feedback_type
    p.dimensions = frozenset(cls.dimensions)
    p.sentiment, p.severity, p.priority = assess.sentiment, assess.severity, assess.priority
    p.action = decide.action
    p.escalation = "yes" if decide.escalation_candidate else "no"
    p.ticket = "yes" if decide.ticket is not None else "no"
    p.ticket_json = _ticket_json(r, decide.action, decide.escalation_candidate, decide.ticket, list(cls.dimensions))
    p.evidence = list(getattr(cls, "evidence", []) or (decide.ticket.evidence if decide.ticket else []))


def single_llm(llm: LLMClient, r: ReviewInput) -> Prediction:
    p = Prediction(system="single_llm", review_id=r.review_id)
    out = llm.structured(ROLE + "\n\n" + knowledge(), _review_block(r) + "\n\nReturn the complete analysis.", SingleLLMOutput)
    _fill(p, r, out, out, out)
    return p


def sequential_pipeline(llm: LLMClient, r: ReviewInput) -> Prediction:
    p = Prediction(system="sequential", review_id=r.review_id)
    sys = ROLE + "\n\n" + knowledge()
    cls = llm.structured(sys, _review_block(r) + "\n\nStep 1 of 3: classify the feedback and extract the issues.", StepClassify)
    assess = llm.structured(sys, _review_block(r) + "\n\nStep 1 result:\n" + cls.model_dump_json(indent=1)
                            + "\n\nStep 2 of 3: assess sentiment, severity and priority.", StepAssess)
    decide = llm.structured(sys, _review_block(r) + "\n\nStep 1 result:\n" + cls.model_dump_json(indent=1)
                            + "\n\nStep 2 result:\n" + assess.model_dump_json(indent=1)
                            + "\n\nStep 3 of 3: decide the action, escalation and ticket.", StepDecide)
    _fill(p, r, cls, assess, decide)
    return p


BASELINES: dict[str, Callable[[LLMClient, ReviewInput], Prediction]] = {
    "single_llm": single_llm,
    "single_llm_cot": single_llm_cot,
    "sequential": sequential_pipeline,
}


def run_baseline(system: str, llm: LLMClient, reviews: list[ReviewInput], reference_date: Optional[date] = None,
                 progress: Optional[Callable[[int, int], None]] = None) -> list[Prediction]:
    """Run one baseline over the reviews. Failures are recorded as FAILED, never filled in."""
    fn = BASELINES[system]
    out = []
    for k, r in enumerate(reviews):
        i0, o0, c0 = llm.usage_snapshot()
        t0 = time.perf_counter()
        if not llm.available:
            p = Prediction(system=system, review_id=r.review_id, error="No LLM available (offline): baseline not run")
        else:
            try:
                p = fn(llm, r)
            except LLMError as e:
                p = Prediction(system=system, review_id=r.review_id, error=str(e)[:300])
        p.seconds = time.perf_counter() - t0
        i1, o1, c1 = llm.usage_snapshot()
        p.input_tokens, p.output_tokens, p.llm_calls = i1 - i0, o1 - o0, c1 - c0
        out.append(p)
        if progress:
            progress(k + 1, len(reviews))
    return out
