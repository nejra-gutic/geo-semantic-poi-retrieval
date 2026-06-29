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

from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from src.retrieval.normalize import normalize
from src.retrieval.intent_classifier import predict
from src.preprocessing.normalize import normalize_text as preprocess_text
from src.retrieval.geo import combine_with_geo, PORTLAND_CENTER
from src.retrieval.hours import is_open_now, is_open_for_query


QUERY_SYNONYMS = {
    "cashpoint":      "atm",
    "cash machine":   "atm",
    "haircut":        "hairdresser",
    "barber":         "hairdresser",
    "hair salon":     "hairdresser",
    "emergency room": "hospital",
    "a&e":            "hospital",
    "vet ":           "veterinary ",
    "chemist":        "pharmacy",
    "drugstore":      "pharmacy",
    "gas station":    "fuel",
    "petrol station": "fuel",
    "bike rental":    "bicycle_rental",
    "car park":       "parking",
    "parkng": "parking",
    "coffeehouse": "coffee shop",
    "ev charging point": "charging_station",
    "ev charger": "charging_station",
    "electric car charging": "charging_station",
}

INTENT_TO_CATEGORY = {
    "find_cafe":      ["cafe", "coffee", "bakery", "bar", "pub"],
    "find_food":      ["restaurant", "fast_food", "food_court", "ice_cream", "deli"],
    "find_service":   ["pharmacy", "doctors", "bank", "hospital", "atm", "clinic",
                       "dentist", "veterinary", "optician", "post_office",
                       "post_depot", "library", "school", "fire_station",
                       "police", "funeral_directors"],
    "find_shop":      ["convenience", "clothes", "supermarket", "furniture", "gift",
                       "hairdresser", "car_repair", "pet", "books", "electronics",
                       "bicycle", "second_hand", "variety_store", "pet_grooming",
                       "hardware", "florist", "jewelry", "shoes", "toys", "sports",
                       "department_store", "mobile_phone", "hotel"],
    "find_transport": [
        "parking", "parking_space", "parking_entrance", "bicycle_parking",
        "bicycle_rental", "charging_station", "fuel", "car_wash",
        "motorcycle_parking", "car_rental", "taxi", "bus_station"
    ],
}

# Words that signal the query is ABOUT a specific category of place.
# If a query contains none of these, hard category filtering is skipped
# (e.g. "who is open this late" has no category signal -> search everything,
# let boolean filters like open_now do the work instead).
CATEGORY_SIGNAL_WORDS = {
    # cafe
    "coffee", "coffe", "cafe", "cafee", "espresso", "latte", "cappuccino",
    "mocha", "flat white", "caffeine", "tea house", "roastery", "bakery",
    "pastry",
    # food
    "restaurant", "restaurants", "food", "eat", "dinner", "lunch",
    "breakfast", "brunch", "pizza", "burger", "sushi", "ramen", "taco",
    "kebab", "bbq", "seafood", "chinese", "thai", "italian", "mexican",
    "indian", "vegan", "fried chicken", "takeout", "takeaway",
    # service
    "pharmacy", "chemist", "hospital", "doctor", "dentist", "bank",
    "bank machine", "atm", "clinic", "veterinary", "vet", "optician",
    "eye doctor", "chiropractor", "physio", "post office", "library",
    "urgent care", "vaccinated", "vaccination",
    # shop
    "shop", "shops", "stores", "store", "buy", "grocery", "supermarket",
    "bookshop", "bookstore", "book", "pet store", "pet shop", "pet food",
    "electronics", "hairdresser", "haircut", "barber", "hair salon",
    "toy", "sneaker", "shoe", "florist", "boutique", "fashion",
    "sports equipment", "hardware",
    # transport
    "parking", "parkng", "transport", "bus", "taxi", "car", "bicycle",
    "bike", "charging", "charger", "fuel", "gas station", "petrol",
    "motorcycle", "vehicle",
}


def has_category_signal(query: str) -> bool:
    """
    Returns True if the query contains at least one word that points to a
    specific category of place. If False, the query is likely a pure
    temporal/generic question (e.g. "who is open this late") with nothing
    for the intent classifier to latch onto -- in that case we should NOT
    apply a hard category filter, since the classifier is just guessing.
    """
    q = query.lower()
    return any(word in q for word in CATEGORY_SIGNAL_WORDS)


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
    "opening_hours",
]


def expand_query_synonyms(query: str) -> str:
    q = query.lower()
    for term, replacement in QUERY_SYNONYMS.items():
        q = q.replace(term, replacement)
    return q


def parse_filters(query: str) -> dict:
    filters = {}
    q = query.lower()
    q = q.replace("opened", "open")

    if any(w in q for w in [
        "wheelchair",
        "accessible",
        "disability",
        "disabled",
        "mobility scooter",
        "step free",
        "step-free",
        "barrier free",
        "no stairs",
        "without stairs",
    ]):
        filters["wheelchair_accessible"] = 1
    if any(w in q for w in ["takeaway", "take away", "takeout"]):
        filters["has_takeaway"] = 1
    if any(w in q for w in ["24/7", "24 7", "open 24", "always open"]):
        filters["is_24_7"] = True
    if any(w in q for w in [
        "open now", "open right now", "still open", "open today",
        "open this", "open late", "open early", "open after",
        "currently open", "open at", "open until",
    ]):
        filters["open_now"] = True

    return filters

TEMPORAL_PHRASES = [
    "open right now", "open now", "still open", "open today",
    "open this late", "open this early", "open this",
    "open late", "open early", "open after", "open until", "open at",
    "currently open", "after midnight", "this evening", "this morning",
    "all night", "right now", "tonight", "this late",
    "24/7", "24 7", "open 24", "always open",
]

WEEKDAY_NAMES = [
    "monday", "tuesday", "wednesday", "thursday",
    "friday", "saturday", "sunday",
]


def extract_temporal_phrase(query: str) -> str:
    """
    Remove temporal/hours phrases from a query, returning the "core" text
    that should be sent to TF-IDF/BM25/embeddings for semantic matching.
    The original (un-stripped) query is still used separately for
    parse_filters() and resolve_check_time(), which need the full phrasing.
    """
    q = query.lower()

    for phrase in TEMPORAL_PHRASES:
        q = q.replace(phrase, " ")

    # "open on <weekday>" / "open <weekday>" -> strip "open" + weekday together
    for day in WEEKDAY_NAMES:
        q = q.replace(f"open on {day}", " ")
        q = q.replace(f"open {day}", " ")
        q = q.replace(day, " ")

    # clean up any leftover standalone "open" that's no longer part of a
    # meaningful phrase (e.g. "pharmacy open" -> "pharmacy")
    q = " ".join(w for w in q.split() if w != "open")

    q = " ".join(q.split())
    return q if q else query  # fallback to original if we stripped everything

def detect_specific_category(query: str):
    q = query.lower()

    if "motorcycle parking" in q:
        return ["motorcycle_parking"]

    if "bicycle rental" in q or "bike rental" in q:
        return ["bicycle_rental"]
    
    if "bike parking" in q or "bicycle parking" in q or "lock my bike" in q or "lock my bicycle" in q:
        return ["bicycle_parking"]
    
    if "bus stop" in q or "bus station" in q:
        return ["bus_station"]

    if "charging" in q or "charger" in q or "electric vehicle" in q or "charging point" in q:
        return ["charging_station"]

    if "parking" in q or "parkng" in q:
        return ["parking", "parking_space", "parking_entrance"]
    
    if "car parts" in q:
        return ["car_parts"]

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

    # MUST come before generic "doctor"/"clinic" check below
    if "optician" in q or "eye doctor" in q or "eye care" in q:
        return ["optician", "doctors", "clinic"]

    if "doctor" in q or "clinic" in q:
        return ["doctors", "clinic"]

    if "haircut" in q or "barber" in q or "hair salon" in q:
        return ["hairdresser"]

    if "vet" in q or "veterinary" in q or "animal" in q or "veterinarian" in q:
        return ["veterinary"]
    
    if "chiropractor" in q or "physio" in q:
        return ["doctors", "clinic"]

    if "bookshop" in q or "bookstore" in q or "book shop" in q or "book" in q:
        return ["books"]

    if "pet store" in q or "pet shop" in q:
        return ["pet", "pet_grooming"]

    if "electronics" in q:
        return ["electronics", "radiotechnics"]

    if "post office" in q or "post depot" in q:
        return ["post_office", "post_depot"]

    if "library" in q:
        return ["library"]

    if "hotel" in q or "lodging" in q or "motel" in q or "hostel" in q:
        return ["hotel"]

    if "school" in q or "university" in q or "college" in q or "campus" in q:
        return ["school", "university", "college"]

    return None


def apply_open_now_filter(results: pd.DataFrame, query: str = "", check_time: datetime = None, boost_pct: float = 0.2, score_col: str = "similarity_score") -> pd.DataFrame:
    """
    Annotate results with 'is_open_now' (True/False/None) and adjust score
    accordingly instead of hard-dropping closed POIs:
      - open (True)    -> +boost_pct * max_score
      - unknown (None) -> no change (don't penalize missing OSM data)
      - closed (False) -> -boost_pct * max_score (penalized, but still
        shown, so the user can see it exists even if currently closed)

    Uses query phrasing to decide WHEN to check (e.g. "open late tonight"
    checks ~22:00, not right now) via is_open_for_query(). If check_time is
    explicitly provided, it overrides this resolution.

    The adjustment is RELATIVE: boost_pct * max score in the pool, so the
    effect is consistent across TF-IDF/Embeddings/Hybrid even though their
    raw score ranges differ. score_col specifies which column to adjust
    (defaults to 'similarity_score'; callers using embeddings/hybrid scores
    should pass 'embedding_score' / 'hybrid_score' / 'combined_score' as
    appropriate).
    """
    if "opening_hours" not in results.columns or results.empty:
        results = results.copy()
        results["is_open_now"] = None
        return results

    results = results.copy()

    if query:
        results["is_open_now"] = results["opening_hours"].apply(
            lambda h: is_open_for_query(h, query, check_time) if pd.notna(h) else None
        )
    else:
        results["is_open_now"] = results["opening_hours"].apply(
            lambda h: is_open_now(h, check_time) if pd.notna(h) else None
        )

    before = len(results)
    print(f"[query] open_now check: {before} POIs annotated (open={int((results['is_open_now']==True).sum())}, "
          f"closed={int((results['is_open_now']==False).sum())}, "
          f"unknown={int(results['is_open_now'].isna().sum())})")

    if score_col in results.columns and not results.empty:
        max_score = results[score_col].max()
        adjustment = max_score * boost_pct if pd.notna(max_score) else 0.0

        def _score_adjustment(v):
            if pd.isna(v):
                return 0.0
            return adjustment if v else -adjustment

        results[score_col] = results[score_col] + results["is_open_now"].apply(_score_adjustment)
        results = results.sort_values(score_col, ascending=False)

    return results


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
    check_time: datetime = None,
) -> pd.DataFrame:
    # Expand synonyms before anything else
    query = expand_query_synonyms(query)

    # Strip temporal phrases before normalization, so TF-IDF/embeddings
    # matching isn't polluted by words like "open"/"now"/"late" (those are
    # still captured separately via parse_filters() + resolve_check_time(),
    # which use the original, un-stripped query).
    query_core = extract_temporal_phrase(query)
    query_norm = normalize(preprocess_text(query_core) or query_core)
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

        CONFIDENCE_THRESHOLD = 40.0

        if specific_categories:
            categories = specific_categories
            print(f"[query] Specific category override: {categories}")
        elif not has_category_signal(query):
            categories = None
            print(f"[query] No category signal detected -> skipping intent category filter, searching all POIs")
        else:
            if confidence >= CONFIDENCE_THRESHOLD:
                categories = INTENT_TO_CATEGORY.get(intent)
                print(f"[query] Intent filter applied: {intent} ({confidence}%)")
            else:
                categories = None
                print(f"[query] Low intent confidence ({confidence}%) -> skipping intent category filter")

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

    # Apply boolean filters (wheelchair, takeaway, is_24_7)
    for col, val in filters.items():
        if col in ("open_now",):
            continue  # handled separately below
        if col in results.columns:
            results = results[results[col] == val]

    # Apply open_now filter (requires parsing opening_hours per row)
    if filters.get("open_now"):
        results = apply_open_now_filter(results, query=query, check_time=check_time, score_col="similarity_score")

    # Keep only relevant columns
    existing_cols = [col for col in RESULT_COLS if col in results.columns]
    extra_cols = [c for c in ["is_open_now"] if c in results.columns]
    results = results[existing_cols + extra_cols + ["similarity_score"]].head(top_k)

    # Apply geo re-ranking if "near me" in query
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