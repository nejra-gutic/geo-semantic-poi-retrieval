"""
retrieval/embeddings.py
-----------------------
Semantic embedding retrieval using Sentence Transformers.

Uses transformer-based embeddings to compare query meaning with POI text meaning.
"""

from pathlib import Path
import numpy as np
import pandas as pd

from sentence_transformers import SentenceTransformer
from src.retrieval.normalize import normalize
from src.preprocessing.normalize import normalize_text as preprocess_text
from src.retrieval.query import expand_query_synonyms, extract_temporal_phrase


MODEL_NAME = "all-MiniLM-L6-v2"


def load_embedding_model(model_name: str = MODEL_NAME) -> SentenceTransformer:
    print(f"[embeddings] Loading model: {model_name}")
    return SentenceTransformer(model_name)


def build_embeddings(
    df: pd.DataFrame,
    model: SentenceTransformer,
    col: str = "poi_text",
    save_path: str = "models/poi_embeddings.npy",
) -> np.ndarray:
    if col not in df.columns:
        print(f"[embeddings] Warning - '{col}' not found, falling back to poi_text")
        col = "poi_text"

    texts = df[col].fillna("").astype(str).tolist()

    print(f"[embeddings] Encoding {len(texts)} POIs...")
    embeddings = model.encode(
        texts,
        batch_size=64,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(save_path, embeddings)

    print(f"[embeddings] Saved embeddings: {save_path}")
    print(f"[embeddings] Shape: {embeddings.shape}")

    return embeddings


def load_embeddings(path: str = "models/poi_embeddings.npy") -> np.ndarray:
    embeddings = np.load(path)
    print(f"[embeddings] Loaded embeddings: {path}")
    print(f"[embeddings] Shape: {embeddings.shape}")
    return embeddings


def get_or_build_embeddings(
    df: pd.DataFrame,
    model: SentenceTransformer,
    col: str = "poi_text",
    path: str = "models/poi_embeddings.npy",
) -> np.ndarray:
    path_obj = Path(path)

    if path_obj.exists():
        return load_embeddings(path)

    return build_embeddings(df, model, col=col, save_path=path)


def search_embeddings(
    query: str,
    model: SentenceTransformer,
    embeddings: np.ndarray,
    df: pd.DataFrame,
    top_k: int = 10,
    intent_model=None,
    intent_vectorizer=None,
    user_lat: float = None,
    user_lon: float = None,
    check_time=None,
    apply_geo: bool = True,
    apply_temporal: bool = True,
) -> pd.DataFrame:
    """
    apply_geo: if False, skips geo re-ranking. Use False when this output
    will be fused into a Hybrid result and geo will be applied once
    after fusion.

    apply_temporal: if False, skips the temporal/open_now boost step.
    Set to False when this function's output is going to be merged into a
    Hybrid result (e.g. in app.py), so the temporal boost is applied exactly
    once -- after the merge -- instead of once here AND once again downstream.
    """
    from src.retrieval.common import (
        apply_intent_boost,
        get_query_core,
        apply_boolean_filters,
        apply_temporal_filter,
        apply_geo_reranking,
    )
    from src.retrieval.query import parse_filters, RESULT_COLS

    if len(df) != len(embeddings):
        raise ValueError(f"[embeddings] df length ({len(df)}) != embeddings length ({len(embeddings)})")

    # 1. Strip temporal phrases before encoding
    query_core = get_query_core(query)
    print(f"[embeddings] Original:   '{query}'")
    print(f"[embeddings] Core:       '{query_core}'")

    # 2. Encode query_core (temporal phrases excluded)
    query_embedding = model.encode(
        [query_core],
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    # 3. Score ALL POIs (soft boost mode)
    print(f"[embeddings] Searching all {len(df)} POIs (soft boost mode)")
    scores = embeddings @ query_embedding[0]

    top_indices = np.argsort(scores)[::-1][:top_k * 5]
    search_df = df.copy()
    if "poi_id" not in search_df.columns:
        search_df["poi_id"] = search_df.index
    search_df = search_df.reset_index(drop=True)

    results = search_df.iloc[top_indices].copy()
    results["embedding_score"] = scores[top_indices]

    print(f"[embeddings] Query: '{query}' -> {len(results)} candidates")

    # 4. Soft boost
    results = apply_intent_boost(
        query, df, results, score_col="embedding_score",
        intent_model=intent_model, intent_vectorizer=intent_vectorizer
    )

    # 5. Boolean filters
    filters = parse_filters(query)
    if filters:
        print(f"[embeddings] Filters detected: {filters}")
    results = apply_boolean_filters(results, filters)

    # Keep only relevant columns
    existing_cols = [col for col in RESULT_COLS if col in results.columns]
    results = results[existing_cols + ["embedding_score"]]

    # 6. GEO FIRST (optional; disable when feeding a Hybrid fusion)
    if apply_geo:
        results = apply_geo_reranking(
            results, query, user_lat, user_lon, score_col="embedding_score"
        )

    # 7. TEMP LAST (skip if this call feeds into a Hybrid merge downstream)
    if apply_temporal:
        score_col = "combined_score" if "combined_score" in results.columns else "embedding_score"
        results = apply_temporal_filter(results, query, filters, check_time=check_time, score_col=score_col)

    results = results.head(top_k)
    print(f"[embeddings] Results found: {len(results)}")
    return results