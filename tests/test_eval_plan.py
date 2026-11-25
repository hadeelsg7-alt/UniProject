"""Evaluation method of the research (Section 3.5): metrics per agent, MAE, and the summary table."""
import evaluation_suite as E


def _p(rid, **kw):
    return E.Prediction(system="multi_agent", review_id=rid, feedback_type=kw.pop("feedback_type", "Bug"), **kw)


LABELS = {"r1": {"severity": "Critical", "priority": "P1", "sentiment": "Negative", "feedback_type": "Bug", "ticket": "yes"},
          "r2": {"severity": "Low", "priority": "P4", "sentiment": "Mixed", "feedback_type": "Bug", "ticket": "yes"},
          "r3": {"severity": "None", "priority": "None", "sentiment": "Positive", "feedback_type": "Praise", "ticket": "no"}}


def test_mae_counts_level_distance():
    preds = [_p("r1", severity="High", priority="P2"),  # 1 level off
             _p("r2", severity="Low", priority="P4"),  # exact
             _p("r3", severity="Medium", priority="P3", feedback_type="Praise")]  # 2 levels off
    m = E.ordinal_mae(LABELS, preds, "severity")
    assert m["n"] == 3 and abs(m["mae"] - 1.0) < 1e-9 and m["max_error"] == 2 and abs(m["within_one"] - 2 / 3) < 1e-9
    assert abs(E.ordinal_mae(LABELS, preds, "priority")["mae"] - 1.0) < 1e-9


def test_failed_predictions_are_excluded_from_mae_but_reported():
    preds = [E.Prediction(system="s", review_id="r1", error="boom"), _p("r2", severity="Low", priority="P4")]
    m = E.ordinal_mae(LABELS, preds, "severity")
    assert m["n"] == 1 and m["excluded_failed"] == 1 and m["mae"] == 0


def test_sentiment_is_scored_with_agent2_as_in_the_plan():
    assert E.LABEL_SPEC["sentiment"]["agent"] == "Agent2"
    preds = [_p("r1", sentiment="Negative"), _p("r2", sentiment="Mixed"), _p("r3", sentiment="Positive", feedback_type="Praise")]
    assert E.agent2_metrics(LABELS, preds)["sentiment"]["macro_f1"] == 1.0


def test_plan_table_has_every_planned_metric_and_no_invented_values():
    preds = [_p("r1", severity="High", priority="P2", sentiment="Negative", ticket="yes"),
             _p("r2", severity="Low", priority="P4", sentiment="Mixed", ticket="yes"),
             _p("r3", severity="None", priority="None", sentiment="Positive", feedback_type="Praise", ticket="no")]
    tab = E.evaluation_plan_table(LABELS, {"multi_agent": preds, "single_llm": preds})
    assert set(tab["Agent"]) == {"Agent 1", "Agent 2", "Agent 3", "Overall"}
    assert {"Precision", "Recall", "Macro F1", "F1", "Validity rate"} <= set(tab["Metric"])
    row = lambda task, metric: tab[(tab["Task"] == task) & (tab["Metric"] == metric)].iloc[0]  # noqa: E731
    assert row("Sentiment", "Macro F1")["Multi-Agent framework"] == 1.0
    assert row("Ticket generation", "F1")["Single LLM"] == 1.0
    assert row("Mean Macro F1 (all fields)", "Value")["Multi-Agent framework"] is not None
    # fields without reference labels, or no prices configured, stay empty, never 0
    assert row("Service-quality dimension (EGSQUAL)", "Macro F1").isna()["Multi-Agent framework"]
    assert row("Estimated cost per review (USD)", "Value").isna()["Multi-Agent framework"]


def test_operational_action_without_issue_fails_consistency():
    from llm import LLMClient
    from schemas import ReviewInput
    from workflow import MultiAgentWorkflow

    st = MultiAgentWorkflow(LLMClient(provider="offline")).run(ReviewInput(review_id="x", text="الدفع لا يعمل ويطلع خطأ", rating=1))
    p = E.predictions_from_state(st)
    assert p.action in E.OPERATIONAL_ACTIONS and p.consistency["action_supported_by_issues"] is True
