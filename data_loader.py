"""Dataset loading for the Arabic government-app reviews CSV.

The uploaded dataset (m-government_apps_Arabic_reviews_.csv, 6,513 rows) was
inspected before writing this module. Its columns are:

    Source, Abbreviation, Content, Rating, Cleaned Content, Stemmed Content,
    Language, Segmented Content

* ``Content``      -> original review text (the agents' main input)
* ``Abbreviation`` -> application identifier (``app_id``)
* ``Source``       -> application name (``app_name``)
* ``Rating``       -> star rating (1-5)

The dataset has NO ground-truth labels for severity, service-quality
dimensions or escalation, and none are assumed. Column detection is flexible,
so a different CSV with similar columns can also be uploaded.
"""
from __future__ import annotations

import io
from pathlib import Path
from typing import Union

import pandas as pd

from arabic_text import normalize
from schemas import ReviewInput

COLUMN_CANDIDATES = {
    "text": ["Content", "content", "review", "Review", "text", "Text", "comment", "review_text"],
    "app_id": ["Abbreviation", "abbreviation", "app_id", "app", "App", "application"],
    "app_name": ["Source", "source", "app_name", "AppName", "application_name"],
    "rating": ["Rating", "rating", "score", "Score", "stars", "Stars"],
}


def _read_csv(source: Union[str, Path, bytes, io.IOBase]) -> pd.DataFrame:
    if isinstance(source, bytes):
        source = io.BytesIO(source)
    last = None
    for enc in ("utf-8-sig", "utf-8", "cp1256", "utf-16"):
        try:
            if hasattr(source, "seek"):
                source.seek(0)
            return pd.read_csv(source, encoding=enc)
        except (UnicodeDecodeError, UnicodeError) as e:
            last = e
    raise ValueError(f"Could not decode CSV as UTF-8/CP1256/UTF-16: {last}")


def detect_columns(df: pd.DataFrame) -> dict:
    mapping = {}
    for key, cands in COLUMN_CANDIDATES.items():
        mapping[key] = next((c for c in cands if c in df.columns), None)
    if mapping["text"] is None:
        raise ValueError(f"No review-text column found. Expected one of {COLUMN_CANDIDATES['text']}; got {list(df.columns)}")
    return mapping


def load_reviews(source: Union[str, Path, bytes, io.IOBase], drop_duplicates: bool = True) -> tuple[pd.DataFrame, dict]:
    """Load and standardise the dataset.

    Returns (df, info). df columns: review_id, original_row, app_id, app_name,
    rating, text, duplicate_count.
    """
    raw = _read_csv(source)
    cols = detect_columns(raw)
    df = pd.DataFrame({
        "original_row": raw.index,
        "text": raw[cols["text"]].astype("string").fillna("").str.strip(),
        "app_id": raw[cols["app_id"]].astype("string").fillna("UNKNOWN").str.strip() if cols["app_id"] else "UNKNOWN",
        "app_name": raw[cols["app_name"]].astype("string").fillna("UNKNOWN").str.strip() if cols["app_name"] else "UNKNOWN",
        "rating": pd.to_numeric(raw[cols["rating"]], errors="coerce").astype("Int64") if cols["rating"] else pd.NA,
    })
    n_raw = len(df)
    empty = df["text"].eq("")
    df = df[~empty].copy()
    df["_norm"] = df["text"].map(normalize)
    df["duplicate_count"] = df.groupby(["app_id", "_norm"])["_norm"].transform("size")
    n_dup = int(df.duplicated(subset=["app_id", "_norm"]).sum())
    if drop_duplicates:
        df = df.drop_duplicates(subset=["app_id", "_norm"], keep="first")
    df = df.drop(columns="_norm")
    df.insert(0, "review_id", df["original_row"].map(lambda i: f"R{int(i):05d}"))
    df = df.reset_index(drop=True)
    info = {
        "rows_in_file": n_raw,
        "empty_text_dropped": int(empty.sum()),
        "duplicates_found_same_app": n_dup,
        "duplicates_dropped": n_dup if drop_duplicates else 0,
        "rows_available": len(df),
        "column_mapping": cols,
        "apps": int(df["app_id"].nunique()),
        "ground_truth_labels": "none (the dataset has no severity / dimension / escalation labels)",
    }
    return df, info


def row_to_review(row: pd.Series) -> ReviewInput:
    rating = row.get("rating")
    return ReviewInput(
        review_id=str(row["review_id"]), text=str(row["text"]), app_id=str(row.get("app_id", "UNKNOWN")),
        app_name=str(row.get("app_name", "UNKNOWN")), rating=None if pd.isna(rating) else int(rating),
        duplicate_count=int(row.get("duplicate_count", 1) or 1),
    )
