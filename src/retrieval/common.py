"""
retrieval/common.py
--------------------
Shared query-processing steps used by ALL retrieval methods (TF-IDF, BM25,
Embeddings) so that intent filtering, boolean filters, and temporal
(open_now) scoring are applied consistently regardless of which scoring
method is used underneath.

Pipeline order (same for every method):
    Query
      -> intent classification + category BOOST (not filter)  (apply_intent_boost)
      -> [retrieval-specific scoring happens here on ALL POIs]
      -> boolean filters (wheelchair, takeaway, 24/7)          (apply_boolean_filters)
      -> geo re-ranking                                         (apply_geo_reranking)
      -> temporal scoring (open_now)                           (apply_temporal_filter)
"""

import pandas as pd
from src.retrieval.intent_classifier import predict
from src.retrieval.query import (
    INTENT_TO_CATEGORY,
    detect_specific_category,
    has_category_signal,
    parse_filters,
    extract_temporal_phrase,
    apply_open_now_filter,
)

CONFIDENCE_THRESHOLD = 40.0
INTENT_BOOST = 0.3  # bonus multiplier for POIs in predicted category


def apply_intent_boost(
    query: str,
    df: pd.DataFrame,
    results: pd.DataFrame,
    score_col: str,
    intent_model=None,
    intent_vectorizer=None,
) -> pd.DataFrame:
    """
    Instead of filtering POIs by intent, boost the score of POIs that
    belong to the predicted category. This way:
    - No POIs are eliminated (zero-overlap impossible)
    - Relevant category POIs still rank higher (boost signal)
    - Classifier errors are recoverable (other POIs still present)

    Boost formula: score += INTENT_BOOST * max_score
    Applied only to POIs whose category_final is in the predicted
    intent's category list (or specific_override list).
    """
    if intent_model is None or intent_vectorizer is None:
        return results

    intent, confidence = predict(query, intent_model, intent_vectorizer)
    print(f"[common] Intent: {intent} ({confidence}%)")

    specific_categories = detect_specific_category(query)

    if specific_categories:
        categories = specific_categories
        print(f"[common] Specific category override: {categories}")
    elif not has_category_signal(query):
        categories = None
        print(f"[common] No category signal -> no boost applied")
    elif confidence >= CONFIDENCE_THRESHOLD:
        categories = INTENT_TO_CATEGORY.get(intent)
        print(f"[common] Intent boost applied: {intent} ({confidence}%)")
    else:
        categories = None
        print(f"[common] Low confidence ({confidence}%) -> no boost applied")

    if categories and score_col in results.columns:
        max_score = results[score_col].max()
        boost_amount = INTENT_BOOST * max_score if pd.notna(max_score) else 0.0

        results = results.copy()
        in_category = results["category_final"].isin(categories)
        results.loc[in_category, score_col] += boost_amount
        results = results.sort_values(score_col, ascending=False)

        print(f"[common] Boosted {in_category.sum()} POIs in categories: {categories}")

    return results


def get_query_core(query: str) -> str:
    """Strip temporal phrases before sending to retrieval scoring."""
    return extract_temporal_phrase(query)


def apply_boolean_filters(results: pd.DataFrame, filters: dict) -> pd.DataFrame:
    """Apply wheelchair_accessible / has_takeaway / is_24_7 filters."""
    for col, val in filters.items():
        if col == "open_now":
            continue
        if col in results.columns:
            results = results[results[col] == val]
    return results


def apply_temporal_filter(
    results: pd.DataFrame,
    query: str,
    filters: dict,
    check_time=None,
    score_col: str = "similarity_score",
) -> pd.DataFrame:
    """Apply open_now soft scoring if query has temporal signal."""
    if filters.get("open_now"):
        results = apply_open_now_filter(
            results, query=query, check_time=check_time, score_col=score_col
        )
    return results


def apply_geo_reranking(
    results: pd.DataFrame,
    query: str,
    user_lat: float,
    user_lon: float,
    score_col: str = "similarity_score",
) -> pd.DataFrame:
    """Apply geo re-ranking if query signals near me/nearby intent."""
    from src.retrieval.geo import combine_with_geo, PORTLAND_CENTER

    near_me = any(
        w in query.lower()
        for w in ["near me", "nearby", "close by", "near downtown"]
    )
    if not near_me:
        return results

    lat = user_lat or PORTLAND_CENTER[0]
    lon = user_lon or PORTLAND_CENTER[1]

    if "latitude" not in results.columns or "longitude" not in results.columns:
        return results

    return combine_with_geo(results, lat, lon, score_col=score_col)