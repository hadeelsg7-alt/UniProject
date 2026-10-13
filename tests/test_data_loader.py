import io

import pytest

from conftest import DATASET
from data_loader import load_reviews, row_to_review


def test_handles_missing_values_and_duplicates():
    csv = ("Source,Abbreviation,Content,Rating\n"
           "App A,AA,التطبيق لا يعمل,1\n"
           "App A,AA,التطبيق لا يعمل ,1\n"      # duplicate (whitespace)
           "App B,BB,التطبيق لا يعمل,2\n"       # same text, different app -> kept
           "App A,AA,,3\n"                       # empty text -> dropped
           "App C,CC,ممتاز,\n")                  # missing rating
    df, info = load_reviews(io.BytesIO(csv.encode("utf-8")))
    assert info["empty_text_dropped"] == 1
    assert info["duplicates_dropped"] == 1
    assert len(df) == 3
    r = row_to_review(df[df.app_id == "CC"].iloc[0])
    assert r.rating is None and r.text == "ممتاز"


def test_missing_text_column_raises():
    with pytest.raises(ValueError):
        load_reviews(io.BytesIO(b"a,b\n1,2\n"))


@pytest.mark.skipif(not DATASET.exists(), reason="dataset not present at data/reviews.csv")
def test_real_dataset_columns():
    df, info = load_reviews(DATASET)
    assert info["rows_in_file"] == 6513
    assert info["column_mapping"] == {"text": "Content", "app_id": "Abbreviation", "app_name": "Source", "rating": "Rating"}
    assert df["review_id"].is_unique
    assert set(df["rating"].dropna().unique()) <= {1, 2, 3, 4, 5}
