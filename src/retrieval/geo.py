"""
retrieval/geo.py
----------------
Geo scoring component for POI retrieval.
Computes distance-based scores to boost nearby POIs.

Usage:
    from src.retrieval.geo import haversine_distance, compute_geo_scores

Default user location: Portland city center (45.5051, -122.6750)
"""

import numpy as np
import pandas as pd
from math import radians, sin, cos, sqrt, atan2


PORTLAND_CENTER = (45.5051, -122.6750)


def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Calculate the great-circle distance between two points in kilometers.
    """
    R = 6371.0  # Earth radius in km

    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))

    return R * c


def compute_geo_scores(
    df: pd.DataFrame,
    user_lat: float,
    user_lon: float,
    decay: float = 0.1,
) -> pd.Series:
    """
    Compute geo scores for all POIs based on distance from user location.
    Score = 1 / (1 + decay * distance_km)
    Closer POIs get higher scores (max 1.0 at distance 0).

    Args:
        df: DataFrame with 'latitude' and 'longitude' columns
        user_lat: User's latitude
        user_lon: User's longitude
        decay: Controls how fast score drops with distance (higher = faster decay)

    Returns:
        Series of geo scores, same index as df
    """
    distances = df.apply(
        lambda row: haversine_distance(user_lat, user_lon, row["latitude"], row["longitude"]),
        axis=1,
    )

    scores = 1.0 / (1.0 + decay * distances)

    return scores


def combine_with_geo(
    results: pd.DataFrame,
    user_lat: float,
    user_lon: float,
    semantic_weight: float = 0.5,
    geo_weight: float = 0.5,
    score_col: str = "embedding_score",
    decay: float = 0.1,
) -> pd.DataFrame:
    """
    Combine semantic scores with geo scores for re-ranking.

    Args:
        results: DataFrame with POI results and semantic scores
        user_lat: User's latitude
        user_lon: User's longitude
        semantic_weight: Weight for semantic score (default 0.5)
        geo_weight: Weight for geo score (default 0.5)
        score_col: Name of semantic score column
        decay: Distance decay factor

    Returns:
        DataFrame re-ranked by combined score
    """
    if "latitude" not in results.columns or "longitude" not in results.columns:
        print("[geo] Warning: latitude/longitude not found, skipping geo scoring")
        return results

    results = results.copy()

    # 1. Semantic score -- NE koristimo min-max po trenutnom skupu kandidata.
    #    Kad su svi kandidati semanticki slicni (npr. svi su "coffee shop" za
    #    query "coffee near me"), min-max razvlaci beznacajnu razliku (npr.
    #    0.616 vs 0.655) na puni 0-1 raspon, pa jedan POI ispadne "los" a drugi
    #    "savrsen" iako su sustinski isti kvalitet matcha. Cosine slicnost je
    #    empirijski uvijek u [0,1] za ovaj model (all-MiniLM-L6-v2) -- vidi
    #    provjeru u chatu: raspon 0.278-0.656 preko razlicitih query-ja, nikad
    #    negativno. Clip je samo defanzivan safety-net, ne stvarna normalizacija.
    sem_scores = results[score_col].clip(lower=0, upper=1)

    # 2. Geo score je vec prirodno u [0,1] (formula 1/(1+decay*dist) to
    #    garantuje: 1.0 na distanci 0, opada ka 0 sa udaljenoscu) -- ne treba
    #    mu dodatna min-max normalizacija iz istog razloga kao gore.
    geo_scores = compute_geo_scores(results, user_lat, user_lon, decay=decay)

    # 3. Kombiniraj
    results["geo_score"] = geo_scores
    results["combined_score"] = semantic_weight * sem_scores + geo_weight * geo_scores

    results = results.sort_values("combined_score", ascending=False)

    print(f"[geo] Re-ranked {len(results)} POIs with geo weight={geo_weight}")

    return results