"""Regression tests on realistic Saudi-dialect reviews.

Two groups:
* WRITTEN: realistic reviews written by the developer (NOT from the dataset) for apps such as
  Absher, Sehhaty, Nafath, Najiz, to test generalisation beyond the scenario set.
* DATASET: real reviews from the dataset on which the offline reasoner previously failed
  (found by blind random sampling), kept as regression cases.
Expectations are the developer's judgement, not verified ground-truth labels.
"""
import pytest

from schemas import ReviewInput

CASES = [
    # (id, app, rating, text, expected primary type, expected overall severity or None, expected action)
    ("W-absher", "Absher", 2, "ابي اجدد الاقامة للعامل والتطبيق يقول حدث خطأ غير متوقع، حاولت اكثر من مره ونفس المشكله، والرسوم انخصمت من الحساب",
     "Bug", "Critical", "Internal technical ticket"),
    ("W-nafath", "Nafath", 1, "الكل عندهم نفس المشكله نفاذ ما يرسل الاشعار للجوال وما نقدر ندخل اي منصة حكومية",
     "Bug", "Critical", "Internal technical ticket"),
    ("W-sehhaty", "Sehhaty", 1, "حجزت موعد في المركز الصحي من التطبيق وطلع لي تم الحجز بس لما رحت قالوا ما فيه موعد باسمك، وفوق كذا التطبيق يعلق كل ما افتح صفحة المواعيد",
     "Bug", "Medium", "Internal technical ticket"),
    ("W-najiz", "Najiz", 3, "كيف اطلع صك الوكالة من التطبيق؟ ما لقيت الخيار", "Inquiry", "Low", "Service improvement request"),
    ("W-tawakkalna", "Tawakkalna", 5, "التطبيق ممتاز وسهل الاستخدام الله يعطيكم العافية", "Praise", None, "No action required"),
    ("W-musaned", "Musaned", 1, "زفت", "General Complaint", None, "Request for additional information"),
    ("R06250", "NAFATH", 1, "عند اضافة البيانات رقم الهويه مع الرقم السري ثم التفعيل يخرج من البرنامج مباشره",
     "Bug", "Medium", "Internal technical ticket"),
    ("R05074", "MOFA", 5, "مشكورين على تسهيل الخدمات لنا", "Praise", None, "No action required"),
    ("R05405", "SAUDIA", 5, "في حجزاتي لا يوجد بها حذف للحجزات المنتهية.", "General Complaint", "Low", "Service improvement request"),
    ("R00252", "NWC", 2, "التطبيق يعلق أثناء طلب الخدمة .. امل تحديثه وإصلاح العطل", "Bug", None, "Internal technical ticket"),
]


@pytest.mark.parametrize("cid,app,rating,text,ftype,sev,action", CASES, ids=[c[0] for c in CASES])
def test_realistic_review(offline_workflow, cid, app, rating, text, ftype, sev, action):
    st = offline_workflow.run(ReviewInput(review_id=cid, text=text, rating=rating, app_id=app, app_name=app))
    assert st.agent1.feedback_type == ftype
    got_sev = st.agent2.overall_severity.value if st.agent2.overall_severity else None
    if sev is not None or ftype in ("Praise",) or cid == "W-musaned":
        assert got_sev == sev
    assert st.agent3.action.value == action
    assert all(i.evidence_verified for i in st.agent1.issues)
