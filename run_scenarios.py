"""Run all test scenarios from the command line and write a report.

    python run_scenarios.py                    # uses LLM_PROVIDER from .env (falls back to offline)
    python run_scenarios.py --provider offline # rule-based reasoner, no API calls
    python run_scenarios.py --only S1 S6

Outputs:
    outputs/scenario_results.json   full shared state of every run
    outputs/scenario_report.md      human-readable report for the supervisor
    outputs/scenario_checks.csv     behavioural evaluation checks
"""
from __future__ import annotations

import argparse
import json
import sys

import config
import evaluation
from llm import LLMClient
from scenarios import get_scenarios, verify_against_dataset
from schemas import WorkflowState
from workflow import MultiAgentWorkflow, state_to_dict


def render_markdown(sc, st: WorkflowState) -> str:
    a1, a2, a3 = st.agent1, st.agent2, st.agent3
    L = [f"## {sc.key} - {sc.title}", "", f"*Purpose:* {sc.purpose}", ""]
    src = "SYNTHETIC (clearly labelled, for demonstration only)" if sc.synthetic else f"Real review from the dataset, row {sc.original_row}"
    L += [f"**Source:** {src} | **App:** {st.review.app_name} ({st.review.app_id}) | **Rating:** {st.review.rating}", "",
          "**Original Arabic review:**", "", f"> {st.review.text.strip()}", "",
          f"**Workflow path:** `{' -> '.join(st.node_trace)}`", "",
          "### Agent 1 - Feedback Investigation",
          f"- Feedback type: **{a1.feedback_type}** (secondary: {', '.join(a1.secondary_types) or '-'})",
          f"- Analysis status: **{a1.status.value}** | ambiguous: {a1.is_ambiguous} {('(' + '; '.join(a1.ambiguity_reasons) + ')') if a1.ambiguity_reasons else ''}",
          f"- Missing information: {', '.join(a1.missing_information) or '-'}",
          f"- Dialect terms: {', '.join(f'{k} = {v}' for k, v in a1.dialect_terms.items()) or '-'}", "",
          "| Issue | Type | EGSQUAL dimension | Evidence (verbatim) | Affected service | Verified |",
          "|---|---|---|---|---|---|"]
    for i in a1.issues:
        L.append(f"| {i.issue_id}: {i.description} | {i.issue_type} | {i.dimension_name} | {i.evidence} | {i.affected_service} | {i.evidence_verified} ({i.evidence_score}) |")
    L += ["", "### Agent 2 - Impact Assessment & Prioritization",
          f"- Overall: **{a2.overall_severity.value if a2.overall_severity else 'no severity'} / {a2.overall_priority.value if a2.overall_priority else '-'}**, verification **{a2.verification_status.value}**",
          f"- Sentiment: {a2.sentiment_intensity} ({a2.sentiment_score}). {a2.sentiment_vs_severity_note}", ""]
    if a2.assessments:
        L += ["| Issue | Service completion | Scope | Risks | Severity | Priority | Rules | Verification |", "|---|---|---|---|---|---|---|---|"]
        for a in a2.assessments:
            L.append(f"| {a.issue_id} | {a.service_completion} | {a.scope} | {', '.join(a.risk_flags) or '-'} | {a.severity.value}{' (prov.)' if a.provisional else ''} | {a.priority.value} | {', '.join(a.rules_applied)} | {a.verification_status.value} |")
        L.append("")
        for a in a2.assessments:
            L.append(f"- {a.issue_id} reasoning: {a.reasoning}")
    L += ["", "### Agent 3 - Resolution & Escalation",
          f"- Actionable: {a3.actionable} | **Action: {a3.action.value}**",
          f"- Rationale: {a3.action_rationale}",
          f"- Secondary actions: {'; '.join(a3.secondary_actions) or '-'}",
          f"- Escalation status: **{a3.escalation.status.value}** | missing fields: {', '.join(a3.escalation.missing_fields) or '-'}"]
    for r in a3.escalation.rules:
        L.append(f"  - {r.rule_id} {r.result}: {r.evidence}")
    if a3.ticket:
        t = a3.ticket
        L += ["", f"**Simulated ticket {t.ticket_id}** ({t.ticket_type})", "",
              f"- Title: {t.title}", f"- Application: {t.application}", f"- Affected service: {t.affected_service}",
              f"- Severity / priority: {t.severity} / {t.priority}", f"- Recommended action: {t.recommended_action}",
              f"- Description: {t.issue_description}", f"- Justification: {t.justification}",
              f"- Human approval required: {t.human_approval_required} | {t.disclaimer}"]
    L += ["", "### Inter-agent revision history"]
    rev = [m for m in st.messages if m.message_type != "HANDOFF"]
    if rev:
        for m in rev:
            L.append(f"- Round {m.round} {m.sender} -> {m.receiver} [{m.concern_code}] {m.issue_ids}: {m.content}")
    else:
        L.append("- No reassessment was needed (Agent 2 found the evidence sufficient).")
    tools_used = {}
    for t in st.tool_calls:
        tools_used.setdefault(t.agent, set()).add(t.tool)
    L += ["", "### Tools used", *[f"- {ag}: {', '.join(sorted(ts))}" for ag, ts in tools_used.items()],
          "", f"### Final operational decision", f"**{a3.final_decision}**", "", "---", ""]
    return "\n".join(L)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--provider", choices=["anthropic", "openai", "offline"], default=None)
    ap.add_argument("--only", nargs="*", help="scenario keys to run, e.g. S1 S6")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)

    llm = LLMClient(provider=args.provider)
    wf = MultiAgentWorkflow(llm)
    scen = [s for s in get_scenarios() if not args.only or s.key in args.only]
    ver = verify_against_dataset()
    print(f"Reasoner: {llm.name}" + (f"  ({llm.note})" if llm.note else ""))
    print(f"Dataset check: {ver}")

    results, md = [], ["# Scenario-based test report", "",
                       f"Reasoner: `{llm.name}`. Escalation and complaint rules are PROPOSED research rules, not official DGA rules. "
                       "All tickets and escalations are SIMULATED.", ""]
    for sc in scen:
        st = wf.run(sc.review, reference_date=sc.reference_date)
        results.append((sc, st))
        md.append(render_markdown(sc, st))
        if not args.quiet:
            print(f"\n=== {sc.key} {sc.title} ===")
            print(f"  review : {sc.review.text.strip()}")
            print(f"  path   : {' -> '.join(st.node_trace)}")
            print(f"  agent1 : {st.agent1.feedback_type} | {st.agent1.status.value} | issues={[(i.issue_id, i.issue_type, i.dimension) for i in st.agent1.issues]}")
            print(f"  agent2 : {[(a.issue_id, a.severity.value, a.priority.value, a.verification_status.value) for a in st.agent2.assessments] or st.agent2.sentiment_vs_severity_note}")
            print(f"  agent3 : {st.agent3.final_decision}")
            for m in st.messages:
                if m.message_type != "HANDOFF":
                    print(f"  msg    : R{m.round} {m.sender}->{m.receiver} [{m.concern_code}] {m.content[:120]}")

    checks = evaluation.checks_table([(sc.key, st, sc.expect) for sc, st in results])
    summ = evaluation.summary(checks)
    md += ["# Behavioural evaluation checks", "",
           f"Passed {summ['checks_passed']} / {summ['applicable']} applicable checks ({summ['not_applicable']} not applicable). "
           "These check rule compliance and evidence support; they are NOT accuracy scores. "
           "Precision/Recall/F1 require verified reference labels (none are included in the dataset).", "",
           checks.to_markdown(index=False) if _has_tabulate() else checks.to_string(index=False)]

    config.OUTPUT_DIR.mkdir(exist_ok=True)
    (config.OUTPUT_DIR / "scenario_report.md").write_text("\n".join(md), encoding="utf-8")
    (config.OUTPUT_DIR / "scenario_results.json").write_text(
        json.dumps([{"scenario": sc.key, "title": sc.title, "synthetic": sc.synthetic, "state": state_to_dict(st)} for sc, st in results],
                   ensure_ascii=False, indent=2), encoding="utf-8")
    checks.to_csv(config.OUTPUT_DIR / "scenario_checks.csv", index=False, encoding="utf-8-sig")
    print(f"\nChecks: {summ}")
    failed = checks[checks["result"] == "FAIL"]
    if not failed.empty:
        print(failed.to_string(index=False))
    print(f"Report written to {config.OUTPUT_DIR / 'scenario_report.md'}")
    return 0 if failed.empty else 1


def _has_tabulate() -> bool:
    try:
        import tabulate  # noqa: F401
        return True
    except ImportError:
        return False


if __name__ == "__main__":
    sys.exit(main())
