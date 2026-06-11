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
from src.retrieval.geo import combine_with_geo, PORTLAND_CENTER


QUERY_SYNONYMS = {
    "cashpoint":     "atm",
    "cash machine":  "atm",
    "haircut":       "hairdresser",
    "barber":        "hairdresser",
    "hair salon":    "hairdresser",
    "emergency room": "hospital",
    "a&e":           "hospital",
    "vet ":          "veterinary ",   
    "bookshop":      "books",
    "book shop":     "books",
}

INTENT_TO_CATEGORY = {
    "find_cafe":      ["cafe", "pub", "bar", "coffee"],
    "find_food":      ["restaurant", "fast_food", "bakery", "food_court", "seafood"],
    "find_service":   ["pharmacy", "doctors", "bank", "hospital", "atm", "clinic",
                       "dentist", "veterinary", "optician"],
    "find_shop":      ["convenience", "clothes", "supermarket", "furniture", "gift",
                       "hairdresser", "car_repair", "pet", "books", "electronics",
                       "bicycle", "second_hand", "variety_store", "pet_grooming"],
    "find_transport": [
        "parking", "parking_space", "parking_entrance", "bicycle_parking",
        "bicycle_rental", "charging_station", "fuel", "car_repair",
        "car_parts", "car_wash", "car_rental", "motorcycle_parking",
        "bicycle_repair_station", "vehicle_inspection", "taxi"
    ],
    "hours_based":    None,
    "accessibility":  ["cafe", "restaurant", "fast_food", "pharmacy", "bar", "pub",
                       "convenience", "clothes", "hospital"],
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
]


def expand_query_synonyms(query: str) -> str:
    q = query.lower()
    for term, replacement in QUERY_SYNONYMS.items():
        q = q.replace(term, replacement)
    return q


def parse_filters(query: str) -> dict:
    filters = {}
    q = query.lower()

    if any(w in q for w in ["wheelchair", "accessible", "disability", "disabled"]):
        filters["wheelchair_accessible"] = 1
    if any(w in q for w in ["takeaway", "take away", "takeout"]):
        filters["has_takeaway"] = 1
    if any(w in q for w in ["24/7", "24 7", "open 24", "always open"]):
        filters["is_24_7"] = True

    return filters


def detect_specific_category(query: str):
    q = query.lower()

    if "bicycle parking" in q or "bike parking" in q:
        return ["bicycle_parking"]

    if "motorcycle parking" in q:
        return ["motorcycle_parking"]

    if "bicycle rental" in q or "bike rental" in q:
        return ["bicycle_rental"]

    if "charging station" in q or "ev charging" in q:
        return ["charging_station"]

    if "parking" in q:
        return ["parking", "parking_space", "parking_entrance"]

    if "cashpoint" in q or "cash machine" in q:
        return ["atm"]

    if "atm" in q:
        return ["atm"]

    if "bank" in q:
        return ["bank"]

    if "pharmacy" in q or "chemist" in q:
        return ["pharmacy"]

    if "hospital" in q or "emergency room" in q:
        return ["hospital"]

    if "dentist" in q or "dental" in q:
        return ["dentist"]

    if "doctor" in q or "clinic" in q:
        return ["doctors", "clinic"]

    if "haircut" in q or "barber" in q or "hair salon" in q:
        return ["hairdresser"]
    
    if "vet" in q or "veterinary" in q or "animal" in q:
        return ["veterinary"]

    if "optician" in q or "eye doctor" in q or "eye care" in q:
        return ["optician", "doctors", "clinic"]

    if "bookshop" in q or "bookstore" in q or "book shop" in q:
        return ["books"]

    if "pet store" in q or "pet shop" in q:
        return ["pet", "pet_grooming"]

    if "electronics" in q:
        return ["electronics", "radiotechnics"]

    return None


def search(
    query: str,
    df: pd.DataFrame,
    vectorizer=None,
    tfidf_matrix=None,
    intent_model=None,
    intent_vectorizer=None,
    top_k: int = 10,
    user_lat: float = None,
    user_lon: float = None,
) -> pd.DataFrame:
    # Expand synonyms before anything else
    query = expand_query_synonyms(query)

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

        specific_categories = detect_specific_category(query)

        if specific_categories:
            categories = specific_categories
            print(f"[query] Specific category override: {categories}")
        else:
            categories = INTENT_TO_CATEGORY.get(intent)

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
        filtered_indices = df_filtered.index.tolist()
        original_indices = df.index.tolist()
        mask = [i for i, idx in enumerate(original_indices) if idx in filtered_indices]

        if not mask:
            print("[query] No POIs after filtering")
            return pd.DataFrame()

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

    # Apply geo re-ranking if location provided or "near me" in query
    near_me = any(w in query.lower() for w in ["near me", "nearby", "close by", "near downtown"])
    if near_me:
        lat = user_lat or PORTLAND_CENTER[0]
        lon = user_lon or PORTLAND_CENTER[1]
        if "latitude" in results.columns and "longitude" in results.columns:
            results = combine_with_geo(
                results,
                lat,
                lon,
                score_col="similarity_score",
            )

    print(f"[query] Results found: {len(results)}")
    return results


def run_interactive(
    df: pd.DataFrame,
    vectorizer=None,
    tfidf_matrix=None,
    intent_model=None,
    intent_vectorizer=None,
) -> None:
    print("\nGeo-Semantic POI Search")
    print("Type your query (or 'exit' to quit)\n")

    while True:
        query = input("Query: ").strip()
        if query.lower() == "exit":
            break
        if not query:
            continue

        results = search(
            query, df, vectorizer, tfidf_matrix, intent_model, intent_vectorizer,
            user_lat=PORTLAND_CENTER[0],
            user_lon=PORTLAND_CENTER[1],
        )
        if results.empty:
            print("No results found.\n")
        else:
            print(results.to_string())
            print()