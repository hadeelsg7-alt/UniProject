"""Scenario definitions for scenario-based testing.

Real scenarios are taken verbatim from the uploaded dataset (identified by
app + original row). The text is embedded too, so scenarios still run when
the dataset file is absent; when it is present, the loader confirms each
review really exists in the file.

`expect` holds the *design expectations* of each scenario (what the scenario
is meant to demonstrate). They are NOT ground-truth labels and are not used
to compute Precision/Recall/F1.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Optional

import pandas as pd

import config
from arabic_text import normalize
from schemas import ActionType, ReviewInput


@dataclass
class Scenario:
    key: str
    title: str
    purpose: str
    review: ReviewInput
    expect: dict = field(default_factory=dict)
    synthetic: bool = False
    original_row: Optional[int] = None
    reference_date: Optional[date] = None


def _real(key, title, purpose, row, app_id, app_name, rating, text, expect):
    return Scenario(key, title, purpose,
                    ReviewInput(review_id=f"R{row:05d}", text=text, app_id=app_id, app_name=app_name, rating=rating),
                    expect, original_row=row)


SCENARIOS: list[Scenario] = [
    _real("S1", "Clear technical issue", "A single, clear failure: the verification code (OTP) is never sent, so login is impossible.",
          705, "Citizen Acc", "Citizen Account", 1,
          "دائماً احاول الدخول عن طريق التطبيق لكن التطبيق لا يرسل رمز التحقق ",
          {"min_issues": 1, "feedback_type": "Bug", "ambiguous": False, "min_severity": "High",
           "actions": [ActionType.INTERNAL_TECHNICAL_TICKET.value]}),
    _real("S2", "Multiple issues in one review", "Three independent issues (crashes, login problem, Arabic UI shows English options) in one review.",
          2464, "SPL", "SPL Online", 1,
          "تطبيق فاشل يقفل كل شوي وتسجيل الدخول مشكله وتختار اللغة العربية لكن جميع الخيارات باللغة الإنجليزية فما ادري ايش الفايدة من اختيار اللغة",
          {"min_issues": 2, "ambiguous": False}),
    _real("S3", "Ambiguous complaint", "'Does not work' - a failure without saying what failed; triggers Agent1<->Agent2 review and a clarification request.",
          1182, "Baladia Report", "My Reports (940 Reports)", 1, "لا يعمل",
          {"ambiguous": True, "revision": True, "actions": [ActionType.REQUEST_ADDITIONAL_INFO.value]}),
    _real("S3b", "Ambiguous 1-star complaint with no operational issue", "Strongly negative ('you fail at everything') but no concrete issue: a 1-star review is NOT automatically critical.",
          3067, "KAHRABA", "AL KAHRABA", 1, "فاشلين في كل شي",
          {"ambiguous": True, "no_severity": True, "actions": [ActionType.REQUEST_ADDITIONAL_INFO.value]}),
    _real("S4", "Positive feedback with improvement suggestion", "Praise plus a request to support fingerprint login.",
          786, "GOSI", "General Organization for Social Insurance", 4,
          "التطبيق مميز واتمنى يدعم تسجيل الدخول عن طريق البصمه",
          {"feedback_type": "Praise", "issue_types": ["Suggestion"], "max_severity": "Low",
           "actions": [ActionType.SERVICE_IMPROVEMENT_REQUEST.value]}),
    _real("S5", "High-impact service issue", "Fees were paid but registration was not completed (financial impact), plus two lower-impact issues.",
          77, "SCE", "Saudi Council of Engineers", 1,
          "البيانات لا تتحدث ولا يوجد خانة للشكوى \nسددت الرسوم ولم يتم التسجيل ",
          {"min_issues": 2, "min_severity": "High", "max_priority": "P2"}),
    _real("S6", "Conflicting classification (praise + bug)", "5-star praise that also reports an input bug: Agent2 detects the type conflict and Agent1 revises its classification.",
          50, "Watani", "Watani - We hear you", 5,
          "برنامج رائع وللمعلومية عند إختيار اللغه العربيه التطبيق لا يقبل سوى الأرقام العربية ",
          {"revision": True, "feedback_type": "Bug"}),
    _real("S7", "Previous complaint mentioned, details missing", "Real review mentioning complaints with no reference number: escalation must be PENDING VERIFICATION, nothing invented.",
          370, "NWC", "National Water Company", 1,
          "البرنامج يعلق وكل ما رفعت شكوى يقول حدث خطأ ولا يقبل تسجيل الدخول مع أن المعلومات صحيحة. \nفريق الشكاوى ما يتجاوبون",
          {"min_issues": 2, "escalation": "PENDING_VERIFICATION"}),
    Scenario(
        "S8", "SYNTHETIC - escalation with complaint history and SLA",
        "CLEARLY SYNTHETIC review + synthetic complaint record SYN-2026-0417 (Open - No Response, 5-day SLA) to demonstrate the escalation decision.",
        ReviewInput(review_id="SYN-001", is_synthetic=True, app_id="SYNTHETIC-APP", app_name="Synthetic demo app", rating=1,
                    text="[مراجعة اصطناعية للاختبار] دفعت رسوم الخدمة وانخصم المبلغ من حسابي لكن الطلب لم يتم، "
                         "ورفعت شكوى رقم SYN-2026-0417 قبل أسابيع وما أحد رد عليها"),
        {"escalation": "CANDIDATE_AWAITING_HUMAN_APPROVAL", "min_severity": "High",
         "actions": [ActionType.ESCALATION_CANDIDATE.value]},
        synthetic=True, reference_date=date(2026, 2, 1),
    ),
]


def get_scenarios() -> list[Scenario]:
    return list(SCENARIOS)


def verify_against_dataset(path: Path | str = config.DEFAULT_DATASET_PATH) -> dict:
    """Confirm each real scenario review exists verbatim in the dataset."""
    p = Path(path)
    if not p.exists():
        return {"dataset_found": False, "checked": {}}
    raw = pd.read_csv(p, encoding="utf-8-sig")
    checked = {}
    for s in SCENARIOS:
        if s.synthetic:
            continue
        ok = False
        if s.original_row is not None and s.original_row < len(raw):
            row = raw.iloc[s.original_row]
            ok = (normalize(str(row["Content"])) == normalize(s.review.text)
                  and str(row["Abbreviation"]).strip() == s.review.app_id and int(row["Rating"]) == s.review.rating)
        checked[s.key] = ok
    return {"dataset_found": True, "checked": checked}
