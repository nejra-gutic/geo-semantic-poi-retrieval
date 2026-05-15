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
from sklearn.metrics.pairwise import cosine_similarity


MODEL_NAME = "all-MiniLM-L6-v2"


def load_embedding_model(model_name: str = MODEL_NAME) -> SentenceTransformer:
    print(f"[embeddings] Loading model: {model_name}")
    return SentenceTransformer(model_name)


def build_embeddings(
    df: pd.DataFrame,
    model: SentenceTransformer,
    col: str = "poi_text_lemma",
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
    col: str = "poi_text_lemma",
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
    df_filtered: pd.DataFrame = None,
) -> pd.DataFrame:
    if df_filtered is not None:
        filtered_indices = df_filtered.index.tolist()
        original_indices = df.index.tolist()
        mask = [i for i, idx in enumerate(original_indices) if idx in filtered_indices]

        if not mask:
            print("[embeddings] No POIs after filtering")
            return pd.DataFrame()

        embeddings_subset = embeddings[mask]
        search_df = df_filtered
    else:
        embeddings_subset = embeddings
        search_df = df

    query_embedding = model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    scores = cosine_similarity(query_embedding, embeddings_subset).flatten()

    top_indices = np.argsort(scores)[::-1][:top_k]
    results = search_df.iloc[top_indices].copy()
    results["embedding_score"] = scores[top_indices]

    print(f"[embeddings] Query: '{query}' → {len(results)} results")
    return results