"""
retrieval/query.py
------------------
Query interface for geo-semantic POI search.
Combines intent classification, TF-IDF matching and boolean filters
to return ranked POI results.

Usage:
    from src.retrieval.query import search
    results = search("coffee near burnside", df, vectorizer, tfidf_matrix,
                     intent_model, intent_vectorizer)
"""

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from src.retrieval.normalize import normalize
from src.retrieval.intent_classifier import predict
from src.preprocessing.normalize import normalize_text as preprocess_text



INTENT_TO_CATEGORY = {
    "find_cafe":      ["cafe", "pub", "bar"],
    "find_food":      ["restaurant", "fast_food", "bakery", "food_court"],
    "find_service":   ["pharmacy", "doctors", "bank", "hospital", "atm", "clinic", "dentist"],
    "find_shop":      ["convenience", "clothes", "supermarket", "furniture", "gift"],
    # keep concrete transport categories instead of filtering only by generic "transport"
    "find_transport": [
        "parking", "parking_space", "parking_entrance", "bicycle_parking",
        "bicycle_rental", "charging_station", "fuel", "car_repair",
        "car_parts", "car_wash", "car_rental", "motorcycle_parking",
        "bicycle_repair_station", "vehicle_inspection", "taxi"
    ],
    "hours_based":    None,
    "accessibility":  ["cafe", "restaurant", "fast_food", "pharmacy", "bar", "pub"],
}

RESULT_COLS = [
    "name",
    "category_final",
    "category_group",
    "cuisine_clean",
    "addr:street",
    "latitude",
    "longitude",
    "wheelchair_accessible",
    "has_takeaway",
    "is_24_7",
    "poi_text",
    "poi_text_lemma",
]


def parse_filters(query: str) -> dict:
    """
    Extract boolean filters from user query.
    e.g. "wheelchair accessible coffee" -> {"wheelchair_accessible": 1}
    """
    filters = {}
    q = query.lower()

    if any(w in q for w in ["wheelchair", "accessible", "disability", "disabled"]):
        filters["wheelchair_accessible"] = 1
    if any(w in q for w in ["takeaway", "take away", "takeout"]):
        filters["has_takeaway"] = 1
    if any(w in q for w in ["24/7", "24 7", "open 24", "always open"]):
        filters["is_24_7"] = True

    return filters


def search(
    query: str,
    df: pd.DataFrame,
    vectorizer=None,
    tfidf_matrix=None,
    intent_model=None,
    intent_vectorizer=None,
    top_k: int = 10,
) -> pd.DataFrame:
    """
    Search POI dataset using intent classification + TF-IDF + boolean filters.

    Args:
        query:             user input string
        df:                cleaned POI dataframe with poi_text_lemma column
        vectorizer:        fitted TfidfVectorizer for POI text
        tfidf_matrix:      fitted TF-IDF matrix
        intent_model:      fitted intent classifier
        intent_vectorizer: fitted TF-IDF vectorizer for intent classifier
        top_k:             number of results to return

    Returns:
        DataFrame of top_k matching POIs with similarity scores
    """
    # Normalize query
    query_norm = normalize(preprocess_text(query) or query)
    if not query_norm:
        print("[query] Empty query after normalization")
        return pd.DataFrame()

    print(f"[query] Original:   '{query}'")
    print(f"[query] Normalized: '{query_norm}'")

    # Predict intent and filter by category
    df_filtered = df.copy()
    if intent_model is not None and intent_vectorizer is not None:
        intent, confidence = predict(query, intent_model, intent_vectorizer)
        print(f"[query] Intent: {intent} ({confidence}%)")

        categories = INTENT_TO_CATEGORY.get(intent)

        # hours_based — pokušaj izvući kategoriju iz query-ja direktno
        if intent == "hours_based":
            keyword_to_category = {
                "pharmacy": ["pharmacy"],
                "cafe": ["cafe"],
                "coffee": ["cafe"],
                "restaurant": ["restaurant", "fast_food"],
                "bar": ["bar"],
                "shop": ["convenience", "clothes"],
                "grocery": ["convenience"],
            }
            for keyword, cats in keyword_to_category.items():
                if keyword in query.lower():
                    categories = cats
                    break
                
        if categories:
            df_filtered = df_filtered[df_filtered["category_final"].isin(categories)]
            print(f"[query] Filtered to categories: {categories} ({len(df_filtered)} POIs)")

    # Extract boolean filters
    filters = parse_filters(query)
    if filters:
        print(f"[query] Filters detected: {filters}")

    # TF-IDF search
    if vectorizer is not None and tfidf_matrix is not None:
        # Get indices of filtered rows
        filtered_indices = df_filtered.index.tolist()
        original_indices = df.index.tolist()
        mask = [i for i, idx in enumerate(original_indices) if idx in filtered_indices]

        if not mask:
            print("[query] No POIs after filtering")
            return pd.DataFrame()

        # Compute similarity only on filtered subset
        tfidf_subset = tfidf_matrix[mask]
        query_vec = vectorizer.transform([query_norm])
        scores = cosine_similarity(query_vec, tfidf_subset).flatten()

        top_indices = np.argsort(scores)[::-1][:top_k * 5]
        results = df_filtered.iloc[top_indices].copy()
        results["similarity_score"] = scores[top_indices]
    else:
        results = df_filtered.copy()
        results["similarity_score"] = 0.0

    # Apply boolean filters
    for col, val in filters.items():
        if col in results.columns:
            results = results[results[col] == val]

    # Keep only relevant columns
    existing_cols = [col for col in RESULT_COLS if col in results.columns]
    results = results[existing_cols + ["similarity_score"]].head(top_k)

    print(f"[query] Results found: {len(results)}")
    return results


def run_interactive(
    df: pd.DataFrame,
    vectorizer=None,
    tfidf_matrix=None,
    intent_model=None,
    intent_vectorizer=None,
) -> None:
    """
    Run an interactive query loop in the terminal.
    Type 'exit' to quit.
    """
    print("\nGeo-Semantic POI Search")
    print("Type your query (or 'exit' to quit)\n")

    while True:
        query = input("Query: ").strip()
        if query.lower() == "exit":
            break
        if not query:
            continue

        results = search(
            query, df, vectorizer, tfidf_matrix, intent_model, intent_vectorizer
        )
        if results.empty:
            print("No results found.\n")
        else:
            print(results.to_string())
            print()