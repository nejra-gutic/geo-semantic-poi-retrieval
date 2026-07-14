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
    "find_service":   ["pharmacy", "chemist", "doctors", "bank", "hospital", "atm", "clinic",
                       "dentist", "veterinary", "optician", "post_office",
                       "post_depot", "library", "school", "fire_station",
                       "police", "funeral_directors",
                       "kindergarten", "community_centre", "massage",
                       "dry_cleaning", "laundry"],
    "find_shop":      ["convenience", "clothes", "supermarket", "furniture", "gift",
                       "hairdresser", "car_repair", "pet", "books", "electronics",
                       "bicycle", "second_hand", "variety_store", "pet_grooming",
                       "hardware", "florist", "jewelry", "shoes", "toys", "sports",
                       "department_store", "mobile_phone", "hotel",
                       "beauty", "cannabis", "tattoo", "alcohol",
                       "student_accommodation", "tyres"],
    "find_transport": [
        "parking", "parking_space", "parking_entrance", "bicycle_parking",
        "bicycle_rental", "charging_station", "fuel", "car_wash",
        "motorcycle_parking", "car_rental", "taxi", "bus_station"
    ],
    "find_worship": ["place_of_worship"],
    "find_entertainment": ["theatre", "cinema", "museum", "gallery", "attraction", "nightclub"],
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
    # worship
    "church", "mosque", "temple", "synagogue", "worship",
    # entertainment
    "theatre", "theater", "cinema", "movie", "museum", "gallery",
    "nightclub", "club", "attraction",
    # shop dodaci
    "beauty", "salon", "spa", "cannabis", "dispensary", "tattoo",
    "alcohol", "liquor", "tire", "tyre",
    # service dodaci
    "kindergarten", "preschool", "community center", "massage",
    "dry cleaning", "laundry", "laundromat",
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
    "poi_id",
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

    if "cashpoint" in q or "cash machine" in q or "bank machine" in q:
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
    
    # --- NOVO: worship ---
    if "church" in q or "mosque" in q or "temple" in q or "synagogue" in q or "worship" in q or "chapel" in q:
        return ["place_of_worship"]

    # --- NOVO: entertainment ---
    if "theatre" in q or "theater" in q:
        return ["theatre"]

    if "cinema" in q or "movie" in q:
        return ["cinema"]

    if "museum" in q:
        return ["museum"]

    if "gallery" in q or "art exhibit" in q:
        return ["gallery"]

    if "nightclub" in q:
        return ["nightclub"]

    if "attraction" in q:
        return ["attraction"]

    # --- NOVO: beauty/tattoo/cannabis/tyres ---
    if "tattoo" in q:
        return ["tattoo"]

    if "cannabis" in q or "dispensary" in q or "weed" in q:
        return ["cannabis"]

    if "beauty" in q or "salon" in q or "spa" in q:
        return ["beauty"]

    if "tire" in q or "tyre" in q:
        return ["tyres"]

    if "kindergarten" in q or "preschool" in q or "daycare" in q:
        return ["kindergarten"]

    if "community center" in q or "community centre" in q:
        return ["community_centre"]

    if "dry cleaning" in q or "laundromat" in q or "laundry" in q:
        return ["dry_cleaning", "laundry"]

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
    from src.retrieval.common import apply_intent_boost

    # 0. Guard: df mora biti u istom redoslijedu/dužini kao TF-IDF matrica
    if tfidf_matrix is not None and len(df) != tfidf_matrix.shape[0]:
        raise ValueError(
            f"[query] df length ({len(df)}) != tfidf_matrix rows ({tfidf_matrix.shape[0]}) "
            f"— je li df filtriran/sortiran nakon build_tfidf()?"
        )
    
    # Expand synonyms before anything else
    query = expand_query_synonyms(query)

    # Strip temporal phrases before TF-IDF scoring
    query_core = extract_temporal_phrase(query)
    query_norm = normalize(preprocess_text(query_core) or query_core)
    if not query_norm:
        print("[query] Empty query after normalization")
        return pd.DataFrame()

    print(f"[query] Original:   '{query}'")
    print(f"[query] Normalized: '{query_norm}'")

    # Extract boolean filters from original query
    filters = parse_filters(query)
    if filters:
        print(f"[query] Filters detected: {filters}")

    # TF-IDF search on ALL POIs (soft boost mode — no hard filtering)
    print(f"[query] Searching all {len(df)} POIs (soft boost mode)")
    if vectorizer is not None and tfidf_matrix is not None:
        query_vec = vectorizer.transform([query_norm])
        scores = cosine_similarity(query_vec, tfidf_matrix).flatten()

        top_indices = np.argsort(scores)[::-1][:top_k * 5]
        results = df.iloc[top_indices].copy()
        results["similarity_score"] = scores[top_indices]
    else:
        results = df.copy()
        results["similarity_score"] = 0.0

    # Soft boost — boost score for POIs in predicted intent category
    results = apply_intent_boost(
        query, df, results, score_col="similarity_score",
        intent_model=intent_model, intent_vectorizer=intent_vectorizer
    )

    # Apply boolean filters (wheelchair, takeaway, is_24_7)
    for col, val in filters.items():
        if col in ("open_now",):
            continue
        if col in results.columns:
            results = results[results[col] == val]

    # Keep only relevant columns
    existing_cols = [col for col in RESULT_COLS if col in results.columns]
    extra_cols = [c for c in ["is_open_now"] if c in results.columns]
    results = results[existing_cols + extra_cols + ["similarity_score"]]


    # GEO FIRST
    from src.retrieval.common import apply_geo_reranking
    results = apply_geo_reranking(results, query, user_lat, user_lon, score_col="similarity_score")

    # TEMP LAST
    if filters.get("open_now"):
        score_col = "combined_score" if "combined_score" in results.columns else "similarity_score"
        results = apply_open_now_filter(
            results, query=query, check_time=check_time, score_col=score_col
        )

    results = results.head(top_k)
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