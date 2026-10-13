"""Agent 3 - Intelligent Resolution & Escalation Agent.

Decides whether the feedback is actionable and which operational action to
take, checks complaint history and SLA data (when available), applies the
PROPOSED escalation-eligibility rules, and generates a SIMULATED ticket.

Safety: nothing is ever submitted to AMER, DGA or any government system.
Human approval is mandatory before any external escalation. Missing
complaint data results in PENDING_VERIFICATION - never invented values.
"""
from __future__ import annotations

import json
import uuid
from datetime import date

from agents.base import BaseAgent
from schemas import (
    SEVERITY_ORDER, ActionType, Agent3Output, AgentMessage, EscalationAssessment, EscalationStatus,
    LLMTicketText, RuleEvaluation, SimulatedTicket, VerificationStatus, WorkflowState,
)
from tools import complaint_record_model, escalation_rules_kb


class ResolutionEscalationAgent(BaseAgent):
    name = "Agent3"
    role = "Intelligent Resolution & Escalation Agent"
    instructions = """You are Agent 3, the Resolution & Escalation Agent. Using the findings of Agents 1 and 2, write a
concise, factual SIMULATED ticket: an English title (max 12 words), the affected service/feature, and a clear issue
description that cites the beneficiary's evidence. Do not invent complaint numbers, dates, statuses or facts that
are not in the inputs. Return only the requested structured output."""

    def resolve(self, state: WorkflowState, reference_date: date | None = None) -> Agent3Output:
        r, a1, a2 = state.review, state.agent1, state.agent2
        state.log(self.name, "START", "Reviewing findings of Agent1 and Agent2")
        overall_sev = a2.overall_severity.value if a2 and a2.overall_severity else None

        # 1) Prior complaint / SLA checks (conditional chain)
        ref = self.call_tool(state, "complaint_reference_extractor", "Check whether the beneficiary mentions a previous complaint",
                             review_text=r.text)
        record, sla = None, None
        if ref["reference"]:
            hist = self.call_tool(state, "complaint_history_lookup", f"Reference '{ref['reference']}' found - look up its status",
                                  reference=ref["reference"], app_id=r.app_id)
            record = hist.get("record")
            if record:
                sla = self.call_tool(state, "sla_checker", "Complaint record found - check SLA response deadline",
                                     record=record, reference_date=reference_date)
        elif ref["complaint_mentioned"]:
            state.log(self.name, "CHECK", f"Previous complaint mentioned ({ref['mentions']}) but no reference number - cannot verify; will not invent one")

        esc_needed = bool(a1.issues) or ref["complaint_mentioned"]
        if esc_needed:
            er = self.call_tool(state, "escalation_rule_engine", "Evaluate proposed escalation-eligibility rules E1-E5",
                                severity=overall_sev, complaint_record=record, sla=sla, complaint_mentioned=ref["complaint_mentioned"])
            escalation = EscalationAssessment(
                status=EscalationStatus(er["status"]), complaint_mentioned=ref["complaint_mentioned"],
                complaint_reference=ref["reference"], complaint_record=complaint_record_model(record),
                sla_deadline=sla.get("deadline") if sla else None, sla_days_overdue=sla.get("days_overdue") if sla else None,
                rules=[RuleEvaluation(**x) for x in er["rules"]], missing_fields=er["missing_fields"],
                human_approval_required=True,
                note=("Proposed research rules - not official DGA rules. " + escalation_rules_kb()["external_channel_label"]),
            )
        else:
            escalation = EscalationAssessment(status=EscalationStatus.NOT_APPLICABLE, note="No issue and no complaint history - escalation not applicable")

        # 2) Decide the operational action
        action, rationale, secondary, actionable = self._decide(state, escalation)
        state.log(self.name, "DECISION", f"Action: {action.value}. {rationale}")
        if secondary:
            state.log(self.name, "DECISION", "Secondary action(s): " + "; ".join(secondary))

        # 3) Generate and validate the simulated ticket
        ticket = None
        if action != ActionType.NO_ACTION:
            ticket = self._ticket(state, action, rationale, escalation)
            val = self.call_tool(state, "ticket_fields_validator", "Validate that the simulated ticket has all required fields",
                                 ticket=ticket.model_dump())
            if not val["complete"]:
                for f in val["missing_fields"]:
                    setattr(ticket, f, getattr(ticket, f) or "unspecified")
                state.log(self.name, "CHECK", f"Ticket fields {val['missing_fields']} were empty - set to 'unspecified'")

        human = escalation.status in (EscalationStatus.CANDIDATE_AWAITING_HUMAN_APPROVAL, EscalationStatus.PENDING_VERIFICATION)
        final = self._final_text(action, escalation, a2)
        out = Agent3Output(actionable=actionable, action=action, action_rationale=rationale, escalation=escalation, ticket=ticket,
                           secondary_actions=secondary, final_decision=final, human_approval_required=human or bool(ticket),
                           reasoner=self.llm.name if self.llm.available else "offline-rule-based")
        state.agent3 = out
        state.messages.append(AgentMessage(sender=self.name, receiver="Human officer", message_type="HANDOFF", content=final))
        state.log(self.name, "FINAL", final)
        return out

    # ------------------------------------------------------------------ #
    def _lead(self, state):
        a2 = state.agent2
        if not a2 or not a2.assessments:
            return None, None
        lead = sorted(a2.assessments, key=lambda a: (-SEVERITY_ORDER[a.severity.value], a.priority.value))[0]
        issue = next(i for i in state.agent1.issues if i.issue_id == lead.issue_id)
        return lead, issue

    def _decide(self, state, esc: EscalationAssessment):
        a1 = state.agent1
        lead, issue = self._lead(state)
        secondary: list[str] = []
        if lead is None:
            if a1.feedback_type == "Praise" and not a1.is_ambiguous:
                return ActionType.NO_ACTION, "Positive feedback with no operational issue; recorded for satisfaction reporting.", [], False
            return (ActionType.REQUEST_ADDITIONAL_INFO,
                    "No concrete issue can be identified (" + "; ".join(a1.ambiguity_reasons or ["unspecific feedback"]) +
                    "). Ask the beneficiary which service/step failed before any ticket can be acted on.", [], False)

        base_action = self._issue_action(lead.issue_id, state)
        for a in state.agent2.assessments:
            if a.issue_id != lead.issue_id:
                act = self._issue_action(a.issue_id, state)
                label = f"{act.value} for {a.issue_id} ({a.severity.value}/{a.priority.value})"
                if act != base_action and label not in secondary:
                    secondary.append(label)

        if esc.status == EscalationStatus.CANDIDATE_AWAITING_HUMAN_APPROVAL:
            secondary.insert(0, f"{base_action.value} for {lead.issue_id}")
            return (ActionType.ESCALATION_CANDIDATE,
                    f"Rules E1-E4 met: previous complaint {esc.complaint_reference} is '{esc.complaint_record.complaint_status}', "
                    f"SLA deadline {esc.sla_deadline} passed by {esc.sla_days_overdue} day(s), severity {lead.severity.value}. "
                    "Escalation is SIMULATED and requires human approval (E5).", secondary, True)
        if esc.complaint_mentioned:
            secondary.insert(0, f"{base_action.value} for {lead.issue_id}")
            if esc.status == EscalationStatus.PENDING_VERIFICATION:
                secondary.append("Ask the beneficiary for the complaint reference number, submission date and current status")
                why = ("The beneficiary mentions a previous complaint, but its reference/status/SLA cannot be verified "
                       f"(missing: {', '.join(esc.missing_fields)}). Escalation status: PENDING VERIFICATION.")
            else:
                why = "A previous complaint is mentioned, but escalation rules are not met: " + \
                      "; ".join(f"{x.rule_id}={x.result}" for x in esc.rules if x.rule_id != "E5")
            return ActionType.FOLLOW_UP_EXISTING_COMPLAINT, why, secondary, True

        if lead.verification_status == VerificationStatus.INSUFFICIENT_EVIDENCE or a1.is_ambiguous:
            secondary.insert(0, f"Provisional {base_action.value.lower()} for {lead.issue_id} (severity {lead.severity.value}, provisional)")
            return (ActionType.REQUEST_ADDITIONAL_INFO,
                    "The feedback reports a problem but evidence is insufficient to act precisely (missing: "
                    + ", ".join(sorted(set(a1.missing_information + issue.missing_information))[:4]) + ").", secondary, True)
        why = {
            ActionType.INTERNAL_TECHNICAL_TICKET: f"{issue.issue_type} affecting service completion ({lead.service_completion}); route to the technical team.",
            ActionType.SERVICE_IMPROVEMENT_REQUEST: f"{issue.issue_type} with impact {lead.service_completion}; route to the product/service-design owner.",
        }[base_action]
        return base_action, f"Lead issue {lead.issue_id} ({lead.severity.value}/{lead.priority.value}): {why}", secondary, True

    def _issue_action(self, issue_id: str, state) -> ActionType:
        iss = next(i for i in state.agent1.issues if i.issue_id == issue_id)
        a = next(x for x in state.agent2.assessments if x.issue_id == issue_id)
        if iss.issue_type == "Bug" and a.service_completion != "NOT_AFFECTED":
            return ActionType.INTERNAL_TECHNICAL_TICKET
        return ActionType.SERVICE_IMPROVEMENT_REQUEST

    def _ticket(self, state, action: ActionType, rationale: str, esc: EscalationAssessment) -> SimulatedTicket:
        r, a1, a2 = state.review, state.agent1, state.agent2
        lead, issue = self._lead(state)
        if lead is None:
            title, svc = "Clarification needed: unspecific beneficiary feedback", "unspecified"
            desc = f"Beneficiary feedback could not be mapped to a concrete issue. Original text: «{r.text.strip()}»"
        else:
            payload = {"application": r.app_name, "lead_issue": issue.model_dump(include={"description", "issue_type", "evidence", "affected_service", "dimension"}),
                       "assessment": lead.model_dump(include={"severity", "priority", "service_completion", "scope", "risk_flags", "impact_summary"}),
                       "other_issues": [i.description for i in a1.issues if i.issue_id != issue.issue_id], "action": action.value}
            if self.llm.available:
                txt, _ = self.reason(state, "ticket drafting", json.dumps(payload, ensure_ascii=False, indent=1), LLMTicketText,
                                     lambda: self._template(r, a1, lead, issue))
            else:
                txt = self._template(r, a1, lead, issue)
                state.log(self.name, "RULE_REASONING", "Ticket text drafted from template (offline)")
            title, svc, desc = txt.title, txt.affected_service, txt.issue_description
        ticket_type = {
            ActionType.INTERNAL_TECHNICAL_TICKET: "Internal technical ticket",
            ActionType.SERVICE_IMPROVEMENT_REQUEST: "Service improvement request",
            ActionType.REQUEST_ADDITIONAL_INFO: "Information request to beneficiary",
            ActionType.FOLLOW_UP_EXISTING_COMPLAINT: "Complaint follow-up",
            ActionType.ESCALATION_CANDIDATE: "Escalation candidate (SIMULATED - awaiting human approval)",
        }[action]
        return SimulatedTicket(
            ticket_id=f"SIM-{date.today().strftime('%Y%m%d')}-{uuid.uuid4().hex[:6].upper()}",
            ticket_type=ticket_type, title=title, application=f"{r.app_name} ({r.app_id})", affected_service=svc,
            issue_description=desc,
            egsqual_dimensions=sorted({i.dimension_name or i.dimension for i in a1.issues}),
            severity=(lead.severity.value + (" (provisional)" if lead.provisional else "")) if lead else "Not assessed",
            priority=lead.priority.value if lead else "P4",
            recommended_action=action.value, justification=rationale,
            evidence=[i.evidence for i in a1.issues] or [r.text.strip()],
            escalation_status=esc.status.value,
            human_approval_required=True,  # every simulated ticket is reviewed by a human officer
            original_review_id=r.review_id,
        )

    @staticmethod
    def _template(r, a1, lead, issue) -> LLMTicketText:
        others = [i.description for i in a1.issues if i.issue_id != issue.issue_id]
        desc = (f"{issue.description}. Impact: {lead.impact_summary} ({lead.service_completion}, scope {lead.scope}"
                + (f", risks: {', '.join(lead.risk_flags)}" if lead.risk_flags else "") + f"). Beneficiary evidence: «{issue.evidence}»."
                + (f" Other issues in the same review: {'; '.join(others)}." if others else ""))
        return LLMTicketText(title=f"[{r.app_id}] {issue.description}"[:120], affected_service=issue.affected_service, issue_description=desc)

    @staticmethod
    def _final_text(action: ActionType, esc: EscalationAssessment, a2) -> str:
        sev = f"{a2.overall_severity.value}/{a2.overall_priority.value}" if a2 and a2.overall_severity else "no severity"
        if a2 and a2.verification_status == VerificationStatus.INSUFFICIENT_EVIDENCE:
            sev += ", provisional - evidence insufficient"
        text = f"{action.value} [{sev}]"
        if esc.status == EscalationStatus.CANDIDATE_AWAITING_HUMAN_APPROVAL:
            text += " - SIMULATED escalation candidate; awaiting mandatory human approval. Nothing has been submitted."
        elif esc.status == EscalationStatus.PENDING_VERIFICATION:
            text += " - escalation PENDING VERIFICATION (complaint details missing; none invented)."
        return text
