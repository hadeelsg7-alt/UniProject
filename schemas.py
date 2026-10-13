"""Pydantic schemas for structured outputs, inter-agent messages and shared state.

Two groups of models:
1. ``LLM*`` models - the small structured outputs the LLM (or the offline
   rule-based reasoner) must return. They are deliberately simple so that
   Claude / OpenAI structured-output modes can enforce them.
2. Agent output / workflow-state models - what the agents produce after
   applying their tools, verification and decision rules.
"""
from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Literal, Optional

from pydantic import BaseModel, Field

# --------------------------------------------------------------------------- #
# Enumerations
# --------------------------------------------------------------------------- #
FeedbackTypeLit = Literal["Bug", "Suggestion", "Inquiry", "Praise", "General Complaint"]
DimensionLit = Literal[
    "EFFICIENCY",
    "SYSTEM_AVAILABILITY",
    "FULFILLMENT",
    "PRIVACY_SECURITY",
    "RESPONSIVENESS",
    "CONTACT",
    "INFORMATION_QUALITY",
]
ServiceCompletionLit = Literal["BLOCKED", "PARTIALLY_BLOCKED", "DEGRADED", "NOT_AFFECTED"]
ScopeLit = Literal["INDIVIDUAL", "RECURRING", "WIDESPREAD"]
RiskFlagLit = Literal["FINANCIAL", "PERSONAL_DATA_SECURITY", "SAFETY", "LEGAL_DEADLINE"]


class AnalysisStatus(str, Enum):
    ACCEPTED = "ACCEPTED"
    REVISED = "REVISED"
    NEEDS_CLARIFICATION = "NEEDS_CLARIFICATION"


class Severity(str, Enum):
    LOW = "Low"
    MEDIUM = "Medium"
    HIGH = "High"
    CRITICAL = "Critical"


SEVERITY_ORDER = {"Low": 1, "Medium": 2, "High": 3, "Critical": 4}


class Priority(str, Enum):
    P1 = "P1"
    P2 = "P2"
    P3 = "P3"
    P4 = "P4"


class VerificationStatus(str, Enum):
    VERIFIED = "VERIFIED"
    VERIFIED_AFTER_REVISION = "VERIFIED_AFTER_REVISION"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"
    NOT_APPLICABLE = "NOT_APPLICABLE"


class ActionType(str, Enum):
    INTERNAL_TECHNICAL_TICKET = "Internal technical ticket"
    SERVICE_IMPROVEMENT_REQUEST = "Service improvement request"
    REQUEST_ADDITIONAL_INFO = "Request for additional information"
    FOLLOW_UP_EXISTING_COMPLAINT = "Follow-up on an existing complaint"
    ESCALATION_CANDIDATE = "Candidate for escalation to AMER / Balaagh Digital (simulated)"
    NO_ACTION = "No action required"


class EscalationStatus(str, Enum):
    NOT_APPLICABLE = "NOT_APPLICABLE"
    NOT_ELIGIBLE = "NOT_ELIGIBLE"
    PENDING_VERIFICATION = "PENDING_VERIFICATION"
    CANDIDATE_AWAITING_HUMAN_APPROVAL = "CANDIDATE_AWAITING_HUMAN_APPROVAL"


# --------------------------------------------------------------------------- #
# Input
# --------------------------------------------------------------------------- #
class ReviewInput(BaseModel):
    review_id: str
    text: str
    app_id: str = "UNKNOWN"
    app_name: str = "UNKNOWN"
    rating: Optional[int] = None
    is_synthetic: bool = False
    duplicate_count: int = 1


# --------------------------------------------------------------------------- #
# LLM structured outputs (also produced by the offline reasoner)
# --------------------------------------------------------------------------- #
class LLMIssue(BaseModel):
    description: str = Field(description="Short English description of ONE independent issue")
    issue_type: FeedbackTypeLit
    dimension: DimensionLit
    evidence_quote: str = Field(description="Exact span copied from the original Arabic review")
    affected_service: str = Field(description="Service/feature affected, or 'unspecified'")
    missing_information: list[str] = Field(default_factory=list)
    confidence: float = Field(ge=0, le=1)


class LLMAgent1Analysis(BaseModel):
    feedback_type: FeedbackTypeLit
    secondary_types: list[FeedbackTypeLit] = Field(default_factory=list)
    issues: list[LLMIssue] = Field(default_factory=list)
    is_ambiguous: bool
    ambiguity_reasons: list[str] = Field(default_factory=list)
    missing_information: list[str] = Field(default_factory=list)
    reasoning: str = ""


class LLMRevisionResponse(BaseModel):
    decision: Literal["CONFIRM", "REVISE"]
    explanation: str
    revised_analysis: LLMAgent1Analysis


class LLMImpact(BaseModel):
    issue_id: str
    service_completion: ServiceCompletionLit
    scope: ScopeLit
    workaround_mentioned: bool
    risk_flags: list[RiskFlagLit] = Field(default_factory=list)
    impact_summary: str
    supporting_quote: str = Field(description="Exact span from the review supporting the impact judgement")
    reasoning: str


class LLMImpactAssessment(BaseModel):
    assessments: list[LLMImpact]


class LLMTicketText(BaseModel):
    title: str
    affected_service: str
    issue_description: str


# --------------------------------------------------------------------------- #
# Logging / communication
# --------------------------------------------------------------------------- #
class LogEntry(BaseModel):
    step: int
    agent: str
    action: str
    detail: str
    tool: Optional[str] = None
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))


class ToolCall(BaseModel):
    agent: str
    tool: str
    reason: str
    input_summary: str
    output_summary: str


class AgentMessage(BaseModel):
    """Inter-agent message stored in the shared state."""

    sender: str
    receiver: str
    message_type: Literal["HANDOFF", "REVIEW_REQUEST", "REVIEW_RESPONSE"]
    round: int = 0
    concern_code: Optional[str] = None
    issue_ids: list[str] = Field(default_factory=list)
    content: str


# --------------------------------------------------------------------------- #
# Agent 1 output
# --------------------------------------------------------------------------- #
class Issue(BaseModel):
    issue_id: str
    description: str
    issue_type: FeedbackTypeLit
    topic: str = ""
    dimension: str
    dimension_name: str = ""
    dimension_definition: str = ""
    evidence: str
    evidence_verified: bool = False
    evidence_score: float = 0.0
    affected_service: str = "unspecified"
    missing_information: list[str] = Field(default_factory=list)
    confidence: float = 0.5
    revision_note: str = ""


class Agent1Output(BaseModel):
    feedback_type: FeedbackTypeLit
    secondary_types: list[str] = Field(default_factory=list)
    issues: list[Issue] = Field(default_factory=list)
    is_ambiguous: bool = False
    ambiguity_reasons: list[str] = Field(default_factory=list)
    missing_information: list[str] = Field(default_factory=list)
    status: AnalysisStatus
    reasoning: str = ""
    dialect_terms: dict[str, str] = Field(default_factory=dict)
    praise_noted: list[str] = Field(default_factory=list)
    revision_round: int = 0
    reasoner: str = "offline"


# --------------------------------------------------------------------------- #
# Agent 2 output
# --------------------------------------------------------------------------- #
class IssueAssessment(BaseModel):
    issue_id: str
    issue_description: str
    service_completion: ServiceCompletionLit
    scope: ScopeLit
    workaround_mentioned: bool = False
    risk_flags: list[str] = Field(default_factory=list)
    impact_summary: str
    severity: Severity
    priority: Priority
    rules_applied: list[str] = Field(default_factory=list)
    reasoning: str
    evidence: list[str] = Field(default_factory=list)
    verification_status: VerificationStatus
    provisional: bool = False


class Agent2Output(BaseModel):
    assessments: list[IssueAssessment] = Field(default_factory=list)
    overall_severity: Optional[Severity] = None
    overall_priority: Optional[Priority] = None
    sentiment_intensity: str = "Neutral"
    sentiment_score: float = 0.0
    sentiment_vs_severity_note: str = ""
    verification_status: VerificationStatus = VerificationStatus.NOT_APPLICABLE
    review_requests_sent: int = 0
    reasoning_summary: str = ""
    reasoner: str = "offline"


# --------------------------------------------------------------------------- #
# Agent 3 output
# --------------------------------------------------------------------------- #
class ComplaintRecord(BaseModel):
    complaint_reference: str
    app_id: str
    issue_summary: str
    submission_date: str
    complaint_status: str
    last_update_date: Optional[str] = None
    sla_response_days: Optional[int] = None
    synthetic: bool = True


class RuleEvaluation(BaseModel):
    rule_id: str
    description: str
    result: Literal["MET", "NOT_MET", "UNKNOWN"]
    evidence: str


class EscalationAssessment(BaseModel):
    status: EscalationStatus
    complaint_mentioned: bool = False
    complaint_reference: Optional[str] = None
    complaint_record: Optional[ComplaintRecord] = None
    sla_deadline: Optional[str] = None
    sla_days_overdue: Optional[int] = None
    rules: list[RuleEvaluation] = Field(default_factory=list)
    missing_fields: list[str] = Field(default_factory=list)
    human_approval_required: bool = True
    note: str = ""


class SimulatedTicket(BaseModel):
    ticket_id: str
    ticket_type: str
    title: str
    application: str
    affected_service: str
    issue_description: str
    egsqual_dimensions: list[str]
    severity: str
    priority: str
    recommended_action: str
    justification: str
    evidence: list[str]
    escalation_status: str
    human_approval_required: bool
    original_review_id: str
    simulated: bool = True
    disclaimer: str = (
        "SIMULATED ticket generated by a research prototype. Nothing has been "
        "submitted to AMER, DGA or any government system."
    )
    created_at: str = Field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))


class Agent3Output(BaseModel):
    actionable: bool
    action: ActionType
    action_rationale: str
    escalation: EscalationAssessment
    ticket: Optional[SimulatedTicket] = None
    secondary_actions: list[str] = Field(default_factory=list)
    final_decision: str
    human_approval_required: bool
    reasoner: str = "offline"


# --------------------------------------------------------------------------- #
# Shared workflow state
# --------------------------------------------------------------------------- #
class WorkflowState(BaseModel):
    review: ReviewInput
    agent1: Optional[Agent1Output] = None
    agent1_history: list[Agent1Output] = Field(default_factory=list)
    agent2: Optional[Agent2Output] = None
    agent3: Optional[Agent3Output] = None
    messages: list[AgentMessage] = Field(default_factory=list)
    decision_log: list[LogEntry] = Field(default_factory=list)
    tool_calls: list[ToolCall] = Field(default_factory=list)
    node_trace: list[str] = Field(default_factory=list)
    revision_count: int = 0
    max_revisions: int = 2
    resolved_concerns: dict[str, str] = Field(default_factory=dict)
    reasoner: str = "offline"
    errors: list[str] = Field(default_factory=list)

    def log(self, agent: str, action: str, detail: str, tool: Optional[str] = None) -> None:
        self.decision_log.append(
            LogEntry(step=len(self.decision_log) + 1, agent=agent, action=action, detail=detail, tool=tool)
        )

    @property
    def revision_occurred(self) -> bool:
        return any(m.message_type == "REVIEW_REQUEST" for m in self.messages)
