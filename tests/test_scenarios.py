"""Scenario-based tests: every scenario must pass all of its behavioural checks."""
import pytest

import evaluation
from conftest import DATASET
from scenarios import get_scenarios, verify_against_dataset

SCENARIOS = get_scenarios()


def test_at_least_five_real_scenarios_and_one_synthetic():
    assert sum(not s.synthetic for s in SCENARIOS) >= 5
    synth = [s for s in SCENARIOS if s.synthetic]
    assert len(synth) == 1 and synth[0].review.is_synthetic and "SYN-" in synth[0].review.text


@pytest.mark.skipif(not DATASET.exists(), reason="dataset not present at data/reviews.csv")
def test_real_scenarios_exist_verbatim_in_dataset():
    ver = verify_against_dataset(DATASET)
    assert ver["dataset_found"] and all(ver["checked"].values()), ver


@pytest.mark.parametrize("sc", SCENARIOS, ids=[s.key for s in SCENARIOS])
def test_scenario_checks_pass(offline_workflow, sc):
    st = offline_workflow.run(sc.review, reference_date=sc.reference_date)
    checks = evaluation.behavioural_checks(st, sc.expect)
    failed = [c for c in checks if c.passed is False]
    assert not failed, failed


def test_prf_only_with_reference_labels(offline_workflow):
    import pandas as pd
    states = [offline_workflow.run(s.review, reference_date=s.reference_date) for s in SCENARIOS[:3]]
    ref = pd.DataFrame({"review_id": [s.review.review_id for s in SCENARIOS[:3]], "feedback_type": ["Bug", "Bug", "Bug"]})
    res = evaluation.prf_from_reference(ref, states)
    assert "feedback_type" in res
    with pytest.raises(ValueError):
        evaluation.prf_from_reference(pd.DataFrame(), states)


def test_multilabel_prf_math():
    t = evaluation._multilabel_prf(["A;B", "A"], ["A", "A;C"])
    micro = t[t.label == "MICRO AVG"].iloc[0]
    assert micro.precision == round(2 / 3, 3) and micro.recall == round(2 / 3, 3)
