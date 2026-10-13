"""Agent orchestration: a lightweight, explicit state machine.

    A1_ANALYZE ──► A2_ASSESS ──(no concerns)──────────────► A3_RESOLVE ──► END
                     ▲   │
                     │   └─(review request & rounds < max)─► A1_REVIEW_RESPONSE
                     └──────────────────────────────────────────┘

* Shared structured state: ``schemas.WorkflowState`` (review, agent outputs,
  inter-agent messages, decision log, tool calls, node trace).
* Transitions are conditional on what the agents decide, not fixed.
* The Agent1 <-> Agent2 revision loop is capped at ``max_revisions`` (default 2).

LangGraph was not used: this explicit machine has no extra dependency and every
transition is visible in ``state.node_trace`` and the decision log.
"""
from __future__ import annotations

from datetime import date
from typing import Iterable, Optional

import config
from agents import FeedbackInvestigationAgent, ImpactAssessmentAgent, ResolutionEscalationAgent
from llm import LLMClient
from schemas import ReviewInput, WorkflowState

A1_ANALYZE = "A1_ANALYZE"
A2_ASSESS = "A2_ASSESS"
A1_REVIEW_RESPONSE = "A1_REVIEW_RESPONSE"
A3_RESOLVE = "A3_RESOLVE"
END = "END"


class MultiAgentWorkflow:
    def __init__(self, llm: Optional[LLMClient] = None, max_revisions: int = config.MAX_REVISIONS):
        self.llm = llm or LLMClient()
        self.max_revisions = max_revisions
        self.agent1 = FeedbackInvestigationAgent(self.llm)
        self.agent2 = ImpactAssessmentAgent(self.llm)
        self.agent3 = ResolutionEscalationAgent(self.llm)

    def run(self, review: ReviewInput, reference_date: Optional[date] = None) -> WorkflowState:
        state = WorkflowState(review=review, max_revisions=self.max_revisions, reasoner=self.llm.name)
        if self.llm.note:
            state.log("Orchestrator", "CONFIG", self.llm.note)
        state.log("Orchestrator", "START", f"Reasoner: {self.llm.name}; max revisions: {self.max_revisions}")
        node, pending = A1_ANALYZE, []
        for _ in range(config.MAX_WORKFLOW_STEPS):
            state.node_trace.append(node)
            if node == A1_ANALYZE:
                self.agent1.analyze(state)
                node = A2_ASSESS
            elif node == A2_ASSESS:
                pending = self.agent2.assess(state)
                if pending:
                    state.messages.extend(pending)
                    node = A1_REVIEW_RESPONSE
                else:
                    node = A3_RESOLVE
            elif node == A1_REVIEW_RESPONSE:
                replies = self.agent1.respond(state, pending)
                state.messages.extend(replies)
                state.revision_count += 1
                state.log("Orchestrator", "TRANSITION", f"Revision round {state.revision_count}/{self.max_revisions} complete -> back to Agent2")
                node = A2_ASSESS
            elif node == A3_RESOLVE:
                self.agent3.resolve(state, reference_date=reference_date)
                node = END
            if node == END:
                state.node_trace.append(END)
                break
        else:  # pragma: no cover - safety guard
            state.errors.append("Workflow step limit reached")
            state.log("Orchestrator", "ERROR", "Workflow step limit reached")
        if state.agent2:
            state.agent2.review_requests_sent = sum(1 for m in state.messages if m.message_type == "REVIEW_REQUEST")
        state.log("Orchestrator", "END", f"Path: {' -> '.join(state.node_trace)}")
        return state

    def run_batch(self, reviews: Iterable[ReviewInput], reference_date: Optional[date] = None, progress=None) -> list[WorkflowState]:
        reviews = list(reviews)
        results = []
        for k, r in enumerate(reviews):
            try:
                results.append(self.run(r, reference_date))
            except Exception as e:  # keep the batch going; record the failure
                st = WorkflowState(review=r)
                st.errors.append(f"{type(e).__name__}: {e}")
                st.log("Orchestrator", "ERROR", f"Run failed: {e}")
                results.append(st)
            if progress:
                progress(k + 1, len(reviews))
        return results


# --------------------------------------------------------------------------- #
# Export helpers
# --------------------------------------------------------------------------- #
def state_to_dict(state: WorkflowState) -> dict:
    return state.model_dump(mode="json")


def state_to_row(state: WorkflowState) -> dict:
    """Flatten one result into a CSV row."""
    a1, a2, a3, r = state.agent1, state.agent2, state.agent3, state.review
    return {
        "review_id": r.review_id, "app_id": r.app_id, "app_name": r.app_name, "rating": r.rating, "review_text": r.text,
        "feedback_type": a1.feedback_type if a1 else None,
        "secondary_types": ";".join(a1.secondary_types) if a1 else None,
        "analysis_status": a1.status.value if a1 else None,
        "is_ambiguous": a1.is_ambiguous if a1 else None,
        "n_issues": len(a1.issues) if a1 else 0,
        "issues": " || ".join(f"{i.issue_id}:{i.description}" for i in a1.issues) if a1 else None,
        "dimensions": ";".join(sorted({i.dimension for i in a1.issues})) if a1 else None,
        "evidence": " || ".join(i.evidence for i in a1.issues) if a1 else None,
        "overall_severity": a2.overall_severity.value if a2 and a2.overall_severity else None,
        "overall_priority": a2.overall_priority.value if a2 and a2.overall_priority else None,
        "sentiment": a2.sentiment_intensity if a2 else None,
        "verification_status": a2.verification_status.value if a2 else None,
        "revision_rounds": state.revision_count,
        "review_requests": sum(1 for m in state.messages if m.message_type == "REVIEW_REQUEST"),
        "action": a3.action.value if a3 else None,
        "escalation_status": a3.escalation.status.value if a3 else None,
        "ticket_id": a3.ticket.ticket_id if a3 and a3.ticket else None,
        "final_decision": a3.final_decision if a3 else None,
        "tools_used": ";".join(sorted({t.tool for t in state.tool_calls})),
        "reasoner": state.reasoner,
        "errors": " | ".join(state.errors),
    }
