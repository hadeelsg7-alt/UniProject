from datetime import date

import tools
from arabic_text import find_cues, normalize


def test_normalization_and_elongation():
    assert normalize("إدارة") == "اداره"
    assert normalize("جداااا") == "جدا"
    assert normalize("مُمتاز") == "ممتاز"


def test_cue_matching_tolerates_prefixes_and_maps_back():
    text = "التطبيق مميز واتمنى يدعم البصمة"
    m = find_cues(text, [normalize("اتمنى")])
    assert m and text[m[0].start:m[0].end] == "واتمنى"


def test_evidence_verifier():
    review = "البرنامج ما يرسل رمز التحقق"
    assert tools.evidence_verifier(review, "ما يرسل رمز التحقق")["verified"]
    assert not tools.evidence_verifier(review, "الدفع لا يعمل")["verified"]
    assert not tools.evidence_verifier(review, "")["verified"]


def test_egsqual_lookup_and_search():
    assert tools.egsqual_kb_lookup("FULFILLMENT")["found"]
    assert not tools.egsqual_kb_lookup("NOT_A_DIMENSION")["found"]
    assert tools.egsqual_kb_search("التطبيق يعلق ولا يعمل")["best_dimension"] == "SYSTEM_AVAILABILITY"


def test_dialect_lookup():
    terms = tools.dialect_lexicon_lookup("التطبيق يعلق كل شوي وما يشتغل")["dialect_terms"]
    assert "يعلق" in terms and "ما يشتغل" in terms


def test_severity_rules():
    f = tools.severity_rule_engine
    assert f("BLOCKED", "INDIVIDUAL", ["FINANCIAL"])["severity"] == "Critical"
    assert f("BLOCKED", "WIDESPREAD", [])["severity"] == "Critical"
    assert f("BLOCKED", "INDIVIDUAL", [])["severity"] == "High"
    assert f("PARTIALLY_BLOCKED", "INDIVIDUAL", ["FINANCIAL"])["severity"] == "High"
    assert f("PARTIALLY_BLOCKED", "INDIVIDUAL", [])["severity"] == "Medium"
    assert f("DEGRADED", "RECURRING", [])["severity"] == "Medium"
    assert f("DEGRADED", "INDIVIDUAL", [])["severity"] == "Low"
    assert f("NOT_AFFECTED", "WIDESPREAD", [])["severity"] == "Low"


def test_impact_urgency_matrix():
    m = tools.impact_urgency_matrix
    assert m("BLOCKED", "INDIVIDUAL", [])["priority"] == "P2"                     # High impact x Medium urgency
    assert m("BLOCKED", "INDIVIDUAL", ["FINANCIAL"])["priority"] == "P1"          # High x High
    assert m("BLOCKED", "WIDESPREAD", [])["priority"] == "P1"
    assert m("PARTIALLY_BLOCKED", "INDIVIDUAL", [])["priority"] == "P3"           # Medium x Medium
    assert m("DEGRADED", "INDIVIDUAL", [])["priority"] == "P4"                    # Low x Low
    assert m("NOT_AFFECTED", "INDIVIDUAL", [])["priority"] == "P4"
    r = m("BLOCKED", "INDIVIDUAL", [], evidence_insufficient=True)
    assert r["priority"] == "P3" and "P-VERIFY" in r["rules"]
    custom = {lvl: {u: "P4" for u in ("High", "Medium", "Low")} for lvl in ("High", "Medium", "Low")}
    assert m("BLOCKED", "WIDESPREAD", ["SAFETY"], matrix=custom)["priority"] == "P4"  # cells are configurable
    assert m("BLOCKED", "INDIVIDUAL", [])["explanation"].startswith("Impact High")


def test_handling_team_and_sla_targets():
    assert tools.handling_team("SYSTEM_AVAILABILITY", "PAYMENT", "Internal technical ticket")["team"] == "Payments & Finance Operations"
    assert tools.handling_team("EFFICIENCY", "", "Service improvement request")["team"] == "Product & User Experience"
    assert "triage" in tools.handling_team("SYSTEM_AVAILABILITY", "", "Request for additional information")["team"]
    t = tools.sla_target("P1")
    assert t["found"] and t["simulated"] and t["response_hours"] < t["resolution_hours"]
    assert tools.sla_target("P2", {"P2": {"response_hours": 1, "resolution_hours": 2}})["resolution_hours"] == 2
    assert not tools.sla_target("P9")["found"]


def test_sentiment_is_separate_from_severity():
    s = tools.sentiment_intensity("سيء جدا جدا فاشل", 1)
    assert s["label"] in ("Strongly negative", "Negative")


def test_complaint_reference_never_invented():
    assert tools.complaint_reference_extractor("التطبيق لا يعمل")["reference"] is None
    r = tools.complaint_reference_extractor("رفعت شكوى ولا احد رد")
    assert r["complaint_mentioned"] and r["reference"] is None
    assert tools.complaint_reference_extractor("شكوى رقم SYN-2026-0417")["reference"] == "SYN-2026-0417"
    # a bare number without a complaint mention is not treated as a complaint reference
    assert tools.complaint_reference_extractor("اتصلت على 920000000")["reference"] is None


def test_complaint_history_and_sla():
    rec = tools.complaint_history_lookup("SYN-2026-0417")["record"]
    sla = tools.sla_checker(rec, reference_date=date(2026, 2, 1))
    assert sla["deadline"] == "2026-01-10" and sla["exceeded"] and sla["days_overdue"] == 22
    assert not tools.complaint_history_lookup("SYN-9999-0000")["found"]
    assert not tools.complaint_history_lookup(None)["found"]


def test_escalation_rules():
    rec = tools.complaint_history_lookup("SYN-2026-0417")["record"]
    sla = tools.sla_checker(rec, reference_date=date(2026, 2, 1))
    assert tools.escalation_rule_engine("Critical", rec, sla, True)["status"] == "CANDIDATE_AWAITING_HUMAN_APPROVAL"
    assert tools.escalation_rule_engine("Low", rec, sla, True)["status"] == "NOT_ELIGIBLE"
    pend = tools.escalation_rule_engine("High", None, None, True)
    assert pend["status"] == "PENDING_VERIFICATION" and "complaint_reference" in pend["missing_fields"]
    assert tools.escalation_rule_engine("Critical", None, None, False)["status"] == "NOT_ELIGIBLE"
    sla_ok = tools.sla_checker(rec, reference_date=date(2026, 1, 7))
    assert tools.escalation_rule_engine("Critical", rec, sla_ok, True)["status"] != "CANDIDATE_AWAITING_HUMAN_APPROVAL"


def test_ticket_validator():
    assert not tools.ticket_fields_validator({"title": "x"})["complete"]


def test_tool_allow_lists():
    assert "Agent3" not in tools.TOOL_REGISTRY["severity_rule_engine"].agents
    assert "Agent1" not in tools.TOOL_REGISTRY["complaint_history_lookup"].agents
