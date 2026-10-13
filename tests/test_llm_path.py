"""Tests of the LLM code path without network access:
1. A stub LLM drives every agent's LLM branch (incl. revision via LLM).
2. The real Anthropic SDK call (messages.parse + structured output) is run
   against a local mock HTTP server, verifying request building and parsing.
3. LLM failures fall back to the rule-based reasoner.
"""
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

import config
from llm import LLMClient, LLMError, StubLLM
from schemas import (LLMAgent1Analysis, LLMImpact, LLMImpactAssessment, LLMIssue, LLMRevisionResponse,
                     LLMTicketText, ReviewInput)
from workflow import MultiAgentWorkflow

REVIEW = ReviewInput(review_id="L1", app_id="TEST", app_name="Test app", rating=5,
                     text="برنامج رائع لكن الدفع لا يعمل ويطلع خطأ")


def _analysis(ftype, quote):
    return LLMAgent1Analysis(feedback_type=ftype, secondary_types=[], is_ambiguous=False, reasoning="stub",
                             issues=[LLMIssue(description="Payment fails with an error", issue_type="Bug", dimension="FULFILLMENT",
                                              evidence_quote=quote, affected_service="Payment", missing_information=[], confidence=0.9)])


def test_stub_llm_drives_all_agents_and_revision():
    impact = LLMImpactAssessment(assessments=[LLMImpact(issue_id="I1", service_completion="BLOCKED", scope="INDIVIDUAL",
                                                       workaround_mentioned=False, risk_flags=["FINANCIAL"],
                                                       impact_summary="Cannot pay", supporting_quote="الدفع لا يعمل", reasoning="stub")])
    stub = StubLLM({
        "LLMAgent1Analysis": [_analysis("Praise", "الدفع لا يعمل ويطلع خطأ")],
        "LLMImpactAssessment": [impact],
        "LLMRevisionResponse": [LLMRevisionResponse(decision="REVISE", explanation="Bug is primary",
                                                    revised_analysis=_analysis("Bug", "الدفع لا يعمل ويطلع خطأ"))],
        "LLMTicketText": [LLMTicketText(title="Payment fails", affected_service="Payment", issue_description="Payment fails with error")],
    })
    st = MultiAgentWorkflow(stub, max_revisions=2).run(REVIEW)
    called = [c[0] for c in stub.calls]
    assert called == ["LLMAgent1Analysis", "LLMImpactAssessment", "LLMRevisionResponse", "LLMTicketText"]
    assert st.agent1_history[0].feedback_type == "Praise" and st.agent1.feedback_type == "Bug"
    assert st.agent2.overall_severity.value == "Critical"  # rules, not the LLM, set severity
    assert st.agent3.ticket.title == "Payment fails"
    assert not st.errors


def test_hallucinated_evidence_is_relocated_or_rejected():
    stub = StubLLM({"LLMAgent1Analysis": [_analysis("Bug", "النص غير موجود في المراجعة اطلاقا")]})
    wf = MultiAgentWorkflow(stub, max_revisions=0)
    st = wf.run(REVIEW)
    # every remaining issue must be supported by the review text
    assert all(i.evidence_verified for i in st.agent1.issues)
    assert any(e.action in ("SELF_REVISION", "REJECT_ISSUE") for e in st.decision_log)


def test_llm_failure_falls_back_to_rules():
    stub = StubLLM({"LLMAgent1Analysis": [LLMError("simulated outage")]})
    st = MultiAgentWorkflow(stub).run(REVIEW)
    assert st.agent3 is not None
    assert any(e.action == "LLM_FALLBACK" for e in st.decision_log)


def test_missing_key_means_offline(monkeypatch):
    monkeypatch.setattr(config, "ANTHROPIC_API_KEY", "")
    c = LLMClient(provider="anthropic")
    assert not c.available and "ANTHROPIC_API_KEY" in c.note


# --------------------------------------------------------------------------- #
# Real Anthropic SDK against a local mock server
# --------------------------------------------------------------------------- #
class _Handler(BaseHTTPRequestHandler):
    requests = []

    def do_POST(self):  # noqa: N802
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        _Handler.requests.append(body)
        payload = {"feedback_type": "Bug", "secondary_types": [], "issues": [], "is_ambiguous": True,
                   "ambiguity_reasons": ["too short"], "missing_information": [], "reasoning": "mock"}
        resp = {"id": "msg_mock", "type": "message", "role": "assistant", "model": body["model"],
                "content": [{"type": "text", "text": json.dumps(payload)}], "stop_reason": "end_turn",
                "stop_sequence": None, "usage": {"input_tokens": 10, "output_tokens": 10}}
        data = json.dumps(resp).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *a):
        pass


@pytest.fixture
def mock_server():
    srv = HTTPServer(("127.0.0.1", 0), _Handler)
    th = threading.Thread(target=srv.serve_forever, daemon=True)
    th.start()
    yield f"http://127.0.0.1:{srv.server_port}"
    srv.shutdown()


def test_anthropic_sdk_structured_output_roundtrip(mock_server, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_BASE_URL", mock_server)
    monkeypatch.setenv("NO_PROXY", "127.0.0.1,localhost")
    monkeypatch.setenv("no_proxy", "127.0.0.1,localhost")
    monkeypatch.setattr(config, "ANTHROPIC_API_KEY", "test-key-not-real")
    client = LLMClient(provider="anthropic")
    assert client.available and client.model == config.ANTHROPIC_MODEL
    out = client.structured("system prompt", "لا يعمل", LLMAgent1Analysis)
    assert isinstance(out, LLMAgent1Analysis) and out.is_ambiguous
    sent = _Handler.requests[-1]
    assert sent["model"] == config.ANTHROPIC_MODEL
    assert sent["output_config"]["format"]["type"] == "json_schema"
    assert "feedback_type" in json.dumps(sent["output_config"]["format"]["schema"])
